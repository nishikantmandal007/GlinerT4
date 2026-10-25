# Load test of one GPU serving glinert4.Detector (production/) behind a batching server: one worker thread owns the
# GPU and takes every request waiting in the queue, up to --max-batch, after collecting for at most --max-wait-ms.
#   closed loop: C clients, each sending its next request the moment the last one returns, for C in --concurrency
#   open loop:   Poisson arrivals at a fixed rate, for each rate (requests/s) in --rates; latency counts queueing
# Requests are texts of the realistic corpus (test_corpus/): its pre-cut chunks (~131 words), or whole documents cut
# into pages of --page-words words (--request page). Per level: requests/min, tokens/min (mDeBERTa tokens of the
# request texts), words/min, latency p50/p95/p99, mean batch size and GPU busy %. --setups can add plain gliner2 fp16
# (quantize=True, Detector(fast=False)) under the same load, as the baseline.
# Backpressure: with --queue-limit the server turns a request away (as an HTTP 429 would) while that many are waiting;
# clients retry up to --retries times with exponential backoff and jitter, and give up after that. Latency includes
# the retries; rejected (turned away at least once) and failed (gave up) are counted per level.
#   python bench/load_test.py --out results/ [--request chunk|page] [--concurrency 1,2,4,...] [--rates 10,50,...]
import argparse
import json
import queue
import random
import statistics
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Busy(Exception):
    pass


class Server:
    def __init__(self, detect, max_batch, max_wait, queue_limit=0):
        self.detect, self.max_batch, self.max_wait, self.queue_limit = detect, max_batch, max_wait, queue_limit
        self.requests = queue.Queue()
        self.batches = []
        self.busy = 0.0
        threading.Thread(target=self.work, daemon=True).start()

    def submit(self, text):
        if self.queue_limit and self.requests.qsize() >= self.queue_limit:
            raise Busy()
        future = Future()
        self.requests.put((text, future))
        return future


    def work(self):
        while True:
            batch = [self.requests.get()]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.requests.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            start = time.perf_counter()
            try:
                results = self.detect([text for text, _ in batch])
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)
            self.busy += time.perf_counter() - start
            self.batches.append(len(batch))


class Client:
    def __init__(self, server, retries, backoff, seed):
        self.server, self.retries, self.backoff, self.rng = server, retries, backoff, random.Random(seed)
        self.rejected = self.failed = 0

    # the request's result, retrying a full queue with exponential backoff and jitter; None when it gives up
    def call(self, text):
        for attempt in range(self.retries + 1):
            try:
                return self.server.submit(text).result()
            except Busy:
                self.rejected += attempt == 0
                if attempt == self.retries:
                    self.failed += 1
                    return None
                time.sleep(self.backoff * 2 ** attempt * self.rng.uniform(0.5, 1.5))


def gpu_sampler(stop, samples):
    import torch

    while not stop.is_set():
        try:
            samples.append(torch.cuda.utilization())
        except Exception:
            return
        time.sleep(0.1)


def measure(server, requests, run):
    server.batches.clear()
    server.busy = 0.0
    stop, samples = threading.Event(), []
    threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    start = time.perf_counter()
    done, clients = run()
    seconds = time.perf_counter() - start
    stop.set()
    rejected, failed = sum(c.rejected for c in clients), sum(c.failed for c in clients)
    done = [(index, latency) for index, latency, ok in done if ok]
    latencies = sorted(latency for _, latency in done) or [0.0]
    served = [requests[index] for index, _ in done]

    def pct(p):
        return round(1000 * latencies[min(len(latencies) - 1, int(p * len(latencies)))], 1)

    attempted = len(done) + failed
    return {"requests": len(done), "seconds": round(seconds, 2),
            "rejected_pct": round(100 * rejected / max(1, attempted), 1), "failed": failed,
            "rpm": round(60 * len(done) / seconds), "tpm": round(60 * sum(r["tokens"] for r in served) / seconds),
            "wpm": round(60 * sum(r["words"] for r in served) / seconds),
            "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99), "mean_ms": round(1000 * statistics.mean(latencies), 1),
            "mean_batch": round(statistics.mean(server.batches), 1) if server.batches else 0,
            "worker_busy": round(server.busy / seconds, 3),
            "gpu_util": round(statistics.mean(samples), 1) if samples else None}


def closed_loop(server, requests, count, duration, retries, backoff):
    done, lock, stop = [], threading.Lock(), time.perf_counter() + duration
    order = iter(range(10 ** 9))
    clients = [Client(server, retries, backoff, seed) for seed in range(count)]

    def loop(client):
        while time.perf_counter() < stop:
            with lock:
                index = next(order) % len(requests)
            start = time.perf_counter()
            ok = client.call(requests[index]["text"]) is not None
            with lock:
                done.append((index, time.perf_counter() - start, ok))

    threads = [threading.Thread(target=loop, args=(client,)) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return done, clients


# every arrival is its own caller (a thread), so a slow or retrying request never holds back the next arrival
def open_loop(server, requests, rate, duration, retries, backoff, seed=0):
    rng, done, lock, threads, clients = random.Random(seed), [], threading.Lock(), [], []
    start, index = time.perf_counter(), 0
    arrival = start

    def call(client, index, arrival):
        ok = client.call(requests[index]["text"]) is not None
        with lock:
            done.append((index, time.perf_counter() - arrival, ok))

    while arrival < start + duration:
        time.sleep(max(0.0, arrival - time.perf_counter()))
        client = Client(server, retries, backoff, index)
        clients.append(client)
        thread = threading.Thread(target=call, args=(client, index % len(requests), arrival))
        thread.start()
        threads.append(thread)
        index += 1
        arrival += rng.expovariate(rate)
    for thread in threads:
        thread.join()
    return done, clients


def load_requests(kind, page_words, tokenizer):
    sys.path.insert(0, str(ROOT / "bench"))
    import realistic_bench

    docs = realistic_bench.prepare(ROOT / "test_corpus", "040")
    if kind == "chunk":
        texts = [chunk for doc in docs for chunk in doc["chunks"]]
    else:
        texts = []
        for doc in docs:
            words = doc["text"].split(" ")
            texts += [" ".join(words[i:i + page_words]) for i in range(0, len(words), page_words)]
    texts = [text for text in texts if text.strip()]
    random.Random(0).shuffle(texts)
    return [{"text": text, "words": len(text.split()),
             "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])} for text in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--request", choices=["chunk", "page"], default="chunk")
    parser.add_argument("--page-words", type=int, default=500)
    parser.add_argument("--setups", default="glinert4", help="glinert4 and/or gliner2_fp16")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--rates", default="", help="open-loop arrival rates in requests/s")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per level")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    parser.add_argument("--queue-limit", type=int, default=0, help="requests waiting before new ones are turned away")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-ms", type=float, default=100.0)
    parser.add_argument("--standin", action="store_true", help="CPU smoke test with a stand-in model")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "production"), str(ROOT / "production" / "tests")]
    import torch

    import glinert4
    from glinert4 import Detector, runtime

    device = "cpu" if args.standin else "cuda"

    def load(setup):
        if args.standin:
            import standin

            return runtime.freeze_eval(standin.build())
        if setup == "glinert4":
            return runtime.load_model("fastino/gliner2-multi-v1", device, 32)
        from gliner2 import AutoExtractor

        return runtime.freeze_eval(AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location=device,
                                                                 quantize=True))

    setups = args.setups.split(",")
    model = load(setups[0])
    requests = load_requests(args.request, args.page_words, model.processor.tokenizer)
    report = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu (stand-in)",
              "request": args.request, "requests_in_pool": len(requests),
              "mean_words": round(statistics.mean(r["words"] for r in requests), 1),
              "mean_tokens": round(statistics.mean(r["tokens"] for r in requests), 1),
              "max_batch": args.max_batch, "max_wait_ms": args.max_wait_ms, "queue_limit": args.queue_limit,
              "retries": args.retries, "backoff_ms": args.backoff_ms, "duration_s": args.duration, "levels": []}
    print("POOL " + json.dumps({k: v for k, v in report.items() if k != "levels"}), flush=True)
    for setup in setups:
        fast = setup == "glinert4"
        glinert4.uninstall()
        if model is None:
            if device == "cuda":
                torch.cuda.empty_cache()
            model = load(setup)
        detector = Detector(model=model, device=device, fast=fast)
        server = Server(detector.detect, args.max_batch, args.max_wait_ms / 1000, args.queue_limit)
        backoff = args.backoff_ms / 1000
        warm = [r["text"] for r in requests[:args.max_batch]]
        detector.detect(warm)
        detector.detect(warm[:1])
        levels = [("closed", int(c)) for c in args.concurrency.split(",") if c]
        levels += [("open", float(r)) for r in args.rates.split(",") if r]
        for mode, level in levels:
            if mode == "closed":
                row = measure(server, requests,
                              lambda: closed_loop(server, requests, level, args.duration, args.retries, backoff))
            else:
                row = measure(server, requests,
                              lambda: open_loop(server, requests, level, args.duration, args.retries, backoff))
            row = {"setup": setup, "mode": mode, "level": level, **row}
            report["levels"].append(row)
            print("LEVEL " + json.dumps(row), flush=True)
        del detector, server
        model = None
    glinert4.uninstall()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_test.json").write_text(json.dumps(report, indent=1))
    lines = [f"Load test on {report['device']}: {args.request} requests (mean {report['mean_words']} words, "
             f"{report['mean_tokens']} tokens), batches up to {args.max_batch}, {args.max_wait_ms} ms collection window, "
             f"{args.duration:.0f} s per level, queue limit {args.queue_limit or 'none'}, {args.retries} retries", "",
             "| setup | load | requests/min | tokens/min | words/min | p50 ms | p95 ms | p99 ms | mean batch | GPU busy "
             "| rejected | failed |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["levels"]:
        load = f"{row['level']} clients" if row["mode"] == "closed" else f"{row['level']:g} req/s offered"
        lines.append(f"| {row['setup']} | {load} | {row['rpm']:,} | {row['tpm']:,} | {row['wpm']:,} | {row['p50_ms']} | "
                     f"{row['p95_ms']} | {row['p99_ms']} | {row['mean_batch']} | "
                     f"{row['gpu_util'] if row['gpu_util'] is not None else '-'}% | {row['rejected_pct']}% | {row['failed']} |")
    (out / "load_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
