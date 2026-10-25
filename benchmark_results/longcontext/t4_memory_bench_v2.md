Document memory on Tesla T4. Finishing = table-cell split and overlap resolution (memory off); the memory columns are the finishing step with the memory on.

| input | words | model s | model s with candidate decode | finishing ms | with memory (default) ms | with memory + pool ms |
|---|---:|---:|---:|---:|---:|---:|
| Contract | 50,805 | 3.082 | 2.755 | 9.3 | 19.8 | 19.9 |
| Realistic corpus 040 | 37,985 | 3.14 | 2.884 | 47.2 | 68.7 | 86.2 |
| Realistic corpus 020 | 18,954 | 1.412 | 1.468 | 21.6 | 42.9 | 38.0 |
| Long document x1 | 37,985 | 3.034 | 2.897 | 51.2 | 68.9 | 70.8 |
| Long document x4 | 151,940 | 12.016 | 12.191 | 264.5 | 302.3 | 402.2 |
| Long document x10 | 379,850 | 31.796 | 36.523 | 535.8 | 839.4 | 1116.8 |

### Contract: 100 pages + header, footer, comments, properties; 50,805 words; 518 ground-truth occurrences; 102 controls

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 0.984 | 1.000 / 0.995 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 6 | 3 | 9.3 |
| labels | 1.000 / 0.984 | 1.000 / 0.995 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 6 | 3 | 11.7 |
| labels+spread | 1.000 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 5 | 3 | 20.0 |
| default (labels+spread+aliases) | 1.000 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 5 | 3 | 19.8 |
| default + bare surnames | 0.997 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 7 | 5 | 3 | 19.7 |
| default + pool | 1.000 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 5 | 3 | 19.9 |

- labels+spread: memory added memory:spread right 1
- default (labels+spread+aliases): memory added memory:spread right 1
- default + bare surnames: memory added memory:alias boundary 1, memory:spread right 1
- default + pool: memory added memory:spread right 1

### Realistic corpus 040: 10 whole documents, 37,985 words

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 47.2 |
| labels | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 52.8 |
| labels+spread | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 68.7 |
| default (labels+spread+aliases) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 68.7 |
| default + bare surnames | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 71.8 |
| default + pool | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.886 | 0.998 / 1.000 | 1 | 48 | 0 | 86.2 |

- default + bare surnames: memory added memory:alias right 2
- default + pool: memory added memory:pooled right 2

### Realistic corpus 020: 10 whole documents, 18,954 words

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 21.6 |
| labels | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 24.9 |
| labels+spread | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 32.1 |
| default (labels+spread+aliases) | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 42.9 |
| default + bare surnames | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 37.3 |
| default + pool | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.886 | 1.000 / 1.000 | 1 | 24 | 0 | 38.0 |


### Long document x1: one document of 37,985 words (~75 pages)

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 0.999 | 1.000 / 1.000 | 0.995 / 0.890 | 0.998 / 1.000 | 3 | 50 | 0 | 51.2 |
| labels | 1.000 / 0.999 | 1.000 / 1.000 | 0.995 / 0.890 | 1.000 / 1.000 | 2 | 50 | 0 | 50.8 |
| labels+spread | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.915 | 1.000 / 1.000 | 2 | 37 | 0 | 79.6 |
| default (labels+spread+aliases) | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.915 | 1.000 / 1.000 | 2 | 37 | 0 | 68.9 |
| default + bare surnames | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.915 | 1.000 / 1.000 | 2 | 37 | 0 | 70.7 |
| default + pool | 1.000 / 1.000 | 1.000 / 1.000 | 0.996 / 0.960 | 1.000 / 1.000 | 2 | 21 | 0 | 70.8 |

- labels: memory added memory:label right 1
- labels+spread: memory added memory:label right 1, memory:spread right 13
- default (labels+spread+aliases): memory added memory:label right 1, memory:spread right 13
- default + bare surnames: memory added memory:label right 1, memory:spread right 13
- default + pool: memory added memory:label right 1, memory:pooled right 19, memory:spread right 95

### Long document x4: one document of 151,940 words (~303 pages)

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 0.999 | 0.999 / 1.000 | 0.999 / 0.892 | 0.997 / 1.000 | 8 | 192 | 0 | 264.5 |
| labels | 1.000 / 0.999 | 0.999 / 1.000 | 0.999 / 0.893 | 1.000 / 1.000 | 3 | 191 | 0 | 233.9 |
| labels+spread | 1.000 / 1.000 | 0.999 / 1.000 | 0.999 / 0.972 | 1.000 / 1.000 | 3 | 56 | 0 | 311.1 |
| default (labels+spread+aliases) | 1.000 / 1.000 | 0.999 / 1.000 | 0.999 / 0.972 | 1.000 / 1.000 | 3 | 56 | 0 | 302.3 |
| default + bare surnames | 1.000 / 1.000 | 0.999 / 1.000 | 0.999 / 0.972 | 1.000 / 1.000 | 3 | 56 | 0 | 554.7 |
| default + pool | 1.000 / 1.000 | 0.999 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 3 | 0 | 0 | 402.2 |

- labels: memory added memory:label right 5
- labels+spread: memory added memory:label right 5, memory:spread right 354
- default (labels+spread+aliases): memory added memory:label right 5, memory:spread right 354
- default + bare surnames: memory added memory:label right 5, memory:spread right 354
- default + pool: memory added memory:label right 5, memory:pooled right 79, memory:spread right 866

### Long document x10: one document of 379,850 words (~759 pages)

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | finishing ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 0.999 | 1.000 / 0.999 | 0.999 / 0.891 | 0.997 / 1.000 | 16 | 482 | 0 | 535.8 |
| labels | 1.000 / 0.999 | 1.000 / 0.999 | 0.999 / 0.892 | 1.000 / 1.000 | 5 | 481 | 0 | 808.7 |
| labels+spread | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.993 | 1.000 / 1.000 | 5 | 39 | 0 | 1108.2 |
| default (labels+spread+aliases) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.993 | 1.000 / 1.000 | 5 | 39 | 0 | 839.4 |
| default + bare surnames | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.993 | 1.000 / 1.000 | 5 | 39 | 0 | 1077.3 |
| default + pool | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 | 7 | 0 | 0 | 1116.8 |

- labels: memory added memory:label right 11
- labels+spread: memory added memory:label right 11, memory:spread right 1937
- default (labels+spread+aliases): memory added memory:alias right 2, memory:label right 11, memory:spread right 1937
- default + bare surnames: memory added memory:alias right 2, memory:label right 11, memory:spread right 1937
- default + pool: memory added memory:alias right 2, memory:label right 11, memory:pooled right 202, memory:pooled wrong label 2, memory:spread right 2205


{
 "device": "Tesla T4",
 "floor": 0.3,
 "contract": {
  "model_s": 3.082,
  "model_with_candidates_s": 2.755,
  "words": 50805,
  "pages": 100,
  "truth_occurrences": 518,
  "controls": 102,
  "not_scored": {
   "Emails": 302,
   "Dates": 1,
   "Zip Code": 2,
   "Phone": 200
  },
  "variants": {
   "off": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 0,
      "fn": 5,
      "precision": 1.0,
      "recall": 0.984
     },
     "Organization": {
      "tp": 201,
      "fp": 0,
      "fn": 1,
      "precision": 1.0,
      "recall": 0.995
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 1,
     "missed": 6,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {},
    "memory_ms": 9.3
   },
   "labels": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 0,
      "fn": 5,
      "precision": 1.0,
      "recall": 0.984
     },
     "Organization": {
      "tp": 201,
      "fp": 0,
      "fn": 1,
      "precision": 1.0,
      "recall": 0.995
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 1,
     "missed": 6,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {},
    "memory_ms": 11.7
   },
   "labels+spread": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 0,
      "fn": 5,
      "precision": 1.0,
      "recall": 0.984
     },
     "Organization": {
      "tp": 202,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 1,
     "missed": 5,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {
     "memory:spread right": 1
    },
    "memory_ms": 20.0
   },
   "default (labels+spread+aliases)": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 0,
      "fn": 5,
      "precision": 1.0,
      "recall": 0.984
     },
     "Organization": {
      "tp": 202,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 1,
     "missed": 5,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {
     "memory:spread right": 1
    },
    "memory_ms": 19.8
   },
   "default + bare surnames": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 1,
      "fn": 5,
      "precision": 0.9968,
      "recall": 0.984
     },
     "Organization": {
      "tp": 202,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 2,
     "missed": 5,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {
     "memory:alias boundary": 1,
     "memory:spread right": 1
    },
    "memory_ms": 19.7
   },
   "default + pool": {
    "per_label": {
     "Person": {
      "tp": 307,
      "fp": 0,
      "fn": 5,
      "precision": 1.0,
      "recall": 0.984
     },
     "Organization": {
      "tp": 202,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     },
     "Street": {
      "tp": 2,
      "fp": 6,
      "fn": 0,
      "precision": 0.25,
      "recall": 1.0
     },
     "City": {
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
     }
    },
    "errors": {
     "boundary": 1,
     "missed": 5,
     "invented": 2,
     "control flagged": 3
    },
    "controls_flagged": 3,
    "memory_added": {
     "memory:spread right": 1
    },
    "memory_ms": 19.9
   }
  }
 },
 "realistic": {
  "040": {
   "model_s": 3.14,
   "model_with_candidates_s": 2.884,
   "documents": 10,
   "words": 37985,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 372,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8857
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 47.2
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 372,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8857
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 52.8
    },
    "labels+spread": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 372,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8857
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 68.7
    },
    "default (labels+spread+aliases)": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 372,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8857
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 68.7
    },
    "default + bare surnames": {
     "per_label": {
      "Person": {
       "tp": 1362,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 372,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8857
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:alias right": 2
     },
     "memory_ms": 71.8
    },
    "default + pool": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 374,
       "fp": 0,
       "fn": 48,
       "precision": 1.0,
       "recall": 0.8863
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:pooled right": 2
     },
     "memory_ms": 86.2
    }
   }
  },
  "020": {
   "model_s": 1.412,
   "model_with_candidates_s": 1.468,
   "documents": 10,
   "words": 18954,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 21.6
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 24.9
    },
    "labels+spread": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 32.1
    },
    "default (labels+spread+aliases)": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 42.9
    },
    "default + bare surnames": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 37.3
    },
    "default + pool": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 187,
       "fp": 1,
       "fn": 24,
       "precision": 0.9947,
       "recall": 0.8863
      },
      "City": {
       "tp": 200,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 38.0
    }
   }
  }
 },
 "long": {
  "x1": {
   "model_s": 3.034,
   "model_with_candidates_s": 2.897,
   "documents": 1,
   "words": 37985,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 0,
       "fn": 2,
       "precision": 1.0,
       "recall": 0.9985
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 387,
       "fp": 2,
       "fn": 48,
       "precision": 0.9949,
       "recall": 0.8897
      },
      "City": {
       "tp": 400,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1,
      "missed": 50
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 51.2
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 0,
       "fn": 2,
       "precision": 1.0,
       "recall": 0.9985
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 388,
       "fp": 2,
       "fn": 48,
       "precision": 0.9949,
       "recall": 0.8899
      },
      "City": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "missed": 50
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 1
     },
     "memory_ms": 50.8
    },
    "labels+spread": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 399,
       "fp": 2,
       "fn": 37,
       "precision": 0.995,
       "recall": 0.9151
      },
      "City": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "missed": 37
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 1,
      "memory:spread right": 13
     },
     "memory_ms": 79.6
    },
    "default (labels+spread+aliases)": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 399,
       "fp": 2,
       "fn": 37,
       "precision": 0.995,
       "recall": 0.9151
      },
      "City": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "missed": 37
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 1,
      "memory:spread right": 13
     },
     "memory_ms": 68.9
    },
    "default + bare surnames": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 399,
       "fp": 2,
       "fn": 37,
       "precision": 0.995,
       "recall": 0.9151
      },
      "City": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "missed": 37
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 1,
      "memory:spread right": 13
     },
     "memory_ms": 70.7
    },
    "default + pool": {
     "per_label": {
      "Person": {
       "tp": 1360,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Street": {
       "tp": 500,
       "fp": 2,
       "fn": 21,
       "precision": 0.996,
       "recall": 0.9597
      },
      "City": {
       "tp": 400,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "missed": 21
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 1,
      "memory:pooled right": 19,
      "memory:spread right": 95
     },
     "memory_ms": 70.8
    }
   }
  },
  "x4": {
   "model_s": 12.016,
   "model_with_candidates_s": 12.191,
   "documents": 1,
   "words": 151940,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 5436,
       "fp": 0,
       "fn": 4,
       "precision": 1.0,
       "recall": 0.9993
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 1558,
       "fp": 2,
       "fn": 188,
       "precision": 0.9987,
       "recall": 0.8923
      },
      "City": {
       "tp": 1600,
       "fp": 5,
       "fn": 0,
       "precision": 0.9969,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 6,
      "missed": 192
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 264.5
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 5437,
       "fp": 0,
       "fn": 3,
       "precision": 1.0,
       "recall": 0.9994
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 1562,
       "fp": 2,
       "fn": 188,
       "precision": 0.9987,
       "recall": 0.8926
      },
      "City": {
       "tp": 1600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1,
      "missed": 191
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 5
     },
     "memory_ms": 233.9
    },
    "labels+spread": {
     "per_label": {
      "Person": {
       "tp": 5440,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 1913,
       "fp": 2,
       "fn": 56,
       "precision": 0.999,
       "recall": 0.9716
      },
      "City": {
       "tp": 1600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1,
      "missed": 56
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 5,
      "memory:spread right": 354
     },
     "memory_ms": 311.1
    },
    "default (labels+spread+aliases)": {
     "per_label": {
      "Person": {
       "tp": 5440,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 1913,
       "fp": 2,
       "fn": 56,
       "precision": 0.999,
       "recall": 0.9716
      },
      "City": {
       "tp": 1600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1,
      "missed": 56
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 5,
      "memory:spread right": 354
     },
     "memory_ms": 302.3
    },
    "default + bare surnames": {
     "per_label": {
      "Person": {
       "tp": 5440,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 1913,
       "fp": 2,
       "fn": 56,
       "precision": 0.999,
       "recall": 0.9716
      },
      "City": {
       "tp": 1600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1,
      "missed": 56
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 5,
      "memory:spread right": 354
     },
     "memory_ms": 554.7
    },
    "default + pool": {
     "per_label": {
      "Person": {
       "tp": 5440,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 1600,
       "fp": 1,
       "fn": 0,
       "precision": 0.9994,
       "recall": 1.0
      },
      "Street": {
       "tp": 2504,
       "fp": 2,
       "fn": 0,
       "precision": 0.9992,
       "recall": 1.0
      },
      "City": {
       "tp": 1600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 2,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 5,
      "memory:pooled right": 79,
      "memory:spread right": 866
     },
     "memory_ms": 402.2
    }
   }
  },
  "x10": {
   "model_s": 31.796,
   "model_with_candidates_s": 36.523,
   "documents": 1,
   "words": 379850,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 13590,
       "fp": 1,
       "fn": 10,
       "precision": 0.9999,
       "recall": 0.9993
      },
      "Organization": {
       "tp": 3997,
       "fp": 1,
       "fn": 3,
       "precision": 0.9997,
       "recall": 0.9992
      },
      "Street": {
       "tp": 3855,
       "fp": 3,
       "fn": 469,
       "precision": 0.9992,
       "recall": 0.8915
      },
      "City": {
       "tp": 4000,
       "fp": 11,
       "fn": 0,
       "precision": 0.9973,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 13,
      "missed": 482
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 535.8
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 13591,
       "fp": 0,
       "fn": 9,
       "precision": 1.0,
       "recall": 0.9993
      },
      "Organization": {
       "tp": 3997,
       "fp": 1,
       "fn": 3,
       "precision": 0.9997,
       "recall": 0.9992
      },
      "Street": {
       "tp": 3865,
       "fp": 3,
       "fn": 469,
       "precision": 0.9992,
       "recall": 0.8918
      },
      "City": {
       "tp": 4000,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 2,
      "missed": 481
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 11
     },
     "memory_ms": 808.7
    },
    "labels+spread": {
     "per_label": {
      "Person": {
       "tp": 13600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 4008,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      },
      "Street": {
       "tp": 5782,
       "fp": 3,
       "fn": 39,
       "precision": 0.9995,
       "recall": 0.9933
      },
      "City": {
       "tp": 4000,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 2,
      "missed": 39
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:label right": 11,
      "memory:spread right": 1937
     },
     "memory_ms": 1108.2
    },
    "default (labels+spread+aliases)": {
     "per_label": {
      "Person": {
       "tp": 13600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 4010,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      },
      "Street": {
       "tp": 5782,
       "fp": 3,
       "fn": 39,
       "precision": 0.9995,
       "recall": 0.9933
      },
      "City": {
       "tp": 4000,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 2,
      "missed": 39
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:alias right": 2,
      "memory:label right": 11,
      "memory:spread right": 1937
     },
     "memory_ms": 839.4
    },
    "default + bare surnames": {
     "per_label": {
      "Person": {
       "tp": 13600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 4010,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      },
      "Street": {
       "tp": 5782,
       "fp": 3,
       "fn": 39,
       "precision": 0.9995,
       "recall": 0.9933
      },
      "City": {
       "tp": 4000,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 2,
      "missed": 39
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:alias right": 2,
      "memory:label right": 11,
      "memory:spread right": 1937
     },
     "memory_ms": 1077.3
    },
    "default + pool": {
     "per_label": {
      "Person": {
       "tp": 13600,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 4010,
       "fp": 1,
       "fn": 0,
       "precision": 0.9998,
       "recall": 1.0
      },
      "Street": {
       "tp": 6252,
       "fp": 3,
       "fn": 0,
       "precision": 0.9995,
       "recall": 1.0
      },
      "City": {
       "tp": 4000,
       "fp": 3,
       "fn": 0,
       "precision": 0.9993,
       "recall": 1.0
      }
     },
     "errors": {
      "invented": 3,
      "wrong label": 4
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:alias right": 2,
      "memory:label right": 11,
      "memory:pooled right": 202,
      "memory:pooled wrong label": 2,
      "memory:spread right": 2205
     },
     "memory_ms": 1116.8
    }
   }
  }
 }
}
