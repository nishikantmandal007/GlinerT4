import itertools
import logging
import os
import queue
import threading
from concurrent.futures import Future

import torch

logger = logging.getLogger(__name__)

HIGH_PRIORITY, LOW_PRIORITY = 0, 1
# texts per GPU run: a high-priority request waits at most one run, about 0.4s at 16 on a T4
RUN_SIZE = int(os.getenv("DETECTION_RUN_SIZE", "16"))
# warm-up texts of increasing length, so every attention kernel size a chunk can need is compiled before traffic
WARM_UP_TEXTS = ["Anna Schmidt works at Siemens AG in Berlin, Hauptstrasse. " * repeat for repeat in (1, 6, 18, 36)]
WARM_UP_LABELS = ["Person", "Organization", "Street", "City", "Insurance Number"]
# runs in flight per GPU, each on its own CUDA stream in this process: while one run waits on the CPU (batch
# preparation, the count head's host copy, decoding) the other keeps the GPU busy; separate processes cannot do this
STREAMS = int(os.getenv("DETECTION_STREAMS", "1"))


def detection_devices():
    configured = os.getenv("DETECTION_DEVICES")
    if configured:
        return [device.strip() for device in configured.split(",") if device.strip()]
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return ["cpu"]


# One queue for every detection request, served by one loop per GPU. A loop takes the most urgent request, adds
# waiting requests with the same labels up to RUN_SIZE texts, and runs them as one batch. Large requests are cut into
# RUN_SIZE slices so urgent requests overtake them between runs; a failed run fails its requests, nothing is dropped.
class DetectionQueue:
    def __init__(self, detect_batch, devices):
        self.detect_batch = detect_batch
        self.pending = queue.PriorityQueue()
        self.arrival = itertools.count()
        # released once per serving thread when its warm-up is done
        self.ready = threading.Semaphore(0)
        for device in devices:
            threading.Thread(target=self.serve, args=(device, 0), daemon=True, name=f"detection-{device}").start()

    def detect(self, texts, labels, priority=LOW_PRIORITY):
        slices = []
        for start in range(0, len(texts), RUN_SIZE):
            done = Future()
            self.pending.put((priority, next(self.arrival), texts[start:start + RUN_SIZE], tuple(labels), done))
            slices.append(done)
        return [found for done in slices for found in done.result()]

    def take_run(self):
        first = self.pending.get()
        run, size, skipped = [first], len(first[2]), []
        while size < RUN_SIZE:
            try:
                waiting = self.pending.get_nowait()
            except queue.Empty:
                break
            if waiting[3] == first[3] and size + len(waiting[2]) <= RUN_SIZE:
                run.append(waiting)
                size += len(waiting[2])
            else:
                skipped.append(waiting)
        for waiting in skipped:
            self.pending.put(waiting)
        return run

    # a GPU's serving threads warm up one after another (each loads its own model copy, see checkpoint), so they
    # never tune the same Triton kernels at once
    def serve(self, device, stream):
        if device != "cpu":
            torch.cuda.set_device(device)
            if STREAMS > 1:
                torch.cuda.set_stream(torch.cuda.Stream(device))
        try:
            self.detect_batch(WARM_UP_TEXTS, WARM_UP_LABELS, device)
            logger.info("Detection on %s (stream %d) is ready", device, stream)
        except Exception:
            logger.exception("Detection warm-up on %s failed; its runs will report the error", device)
        self.ready.release()
        if stream + 1 < STREAMS:
            threading.Thread(target=self.serve, args=(device, stream + 1), daemon=True,
                             name=f"detection-{device}-{stream + 1}").start()
        while True:
            run = self.take_run()
            texts = [text for request in run for text in request[2]]
            try:
                found = self.detect_batch(texts, list(run[0][3]), device)
            except Exception as error:
                logger.exception("Detection run of %d texts on %s failed", len(texts), device)
                for request in run:
                    request[4].set_exception(error)
                continue
            offset = 0
            for request in run:
                request[4].set_result(found[offset:offset + len(request[2])])
                offset += len(request[2])
