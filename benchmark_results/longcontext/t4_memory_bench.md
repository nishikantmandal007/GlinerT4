Document memory on Tesla T4, candidates from a second decode at 0.3

### Contract: 104 parts, 50,805 words, 518 ground-truth occurrences, 102 controls, model 3.6 s

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | memory ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 0.984 | 1.000 / 0.995 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 6 | 3 | 49.3 |
| labels | 1.000 / 0.984 | 1.000 / 0.995 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 6 | 3 | 50.3 |
| labels+pool | 1.000 / 0.984 | 1.000 / 0.995 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 6 | 3 | 49.9 |
| labels+pool+spread | 1.000 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 5 | 3 | 142.7 |
| all four | 0.997 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 7 | 5 | 3 | 1020.9 |
| all four, no bare surnames | 1.000 / 0.984 | 1.000 / 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 6 | 5 | 3 | 1054.6 |

Spans the memory added or relabelled (all four steps), by verdict: memory:alias boundary 1, memory:spread right 1

### Realistic corpus 040: 10 documents, 37,985 words, model 3.541 s

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | memory ms |
|---|---|---|---|---|---|---|---|---|
| off | 0.999 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 345 | 48 | 0 | 115.4 |
| labels | 0.999 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 345 | 48 | 0 | 143.6 |
| labels+pool | 0.999 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 347 | 48 | 0 | 126.7 |
| labels+pool+spread | 0.999 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 347 | 48 | 0 | 184.5 |
| all four | 0.997 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 349 | 48 | 0 | 1165.7 |
| all four, no bare surnames | 0.999 / 1.000 | 0.998 / 1.000 | 0.083 / 0.392 | 0.998 / 1.000 | 347 | 48 | 0 | 1140.6 |

Spans the memory added or relabelled (all four steps), by verdict: memory:alias boundary 2, memory:pooled boundary 2

### Realistic corpus 020: 10 documents, 18,954 words, model 1.678 s

| variant | Person P / R | Organization P / R | Street P / R | City P / R | FP | FN | controls flagged | memory ms |
|---|---|---|---|---|---|---|---|---|
| off | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 60.4 |
| labels | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 37.0 |
| labels+pool | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 38.1 |
| labels+pool+spread | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 68.2 |
| all four | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 619.3 |
| all four, no bare surnames | 1.000 / 1.000 | 0.995 / 1.000 | 0.074 / 0.368 | 1.000 / 1.000 | 175 | 24 | 0 | 602.0 |

Spans the memory added or relabelled (all four steps), by verdict: none


{
 "device": "Tesla T4",
 "floor": 0.3,
 "contract": {
  "model_seconds": 3.6,
  "words": 50805,
  "pages": 104,
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
    "memory_ms": 49.3
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
    "memory_ms": 50.3
   },
   "labels+pool": {
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
    "memory_ms": 49.9
   },
   "labels+pool+spread": {
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
    "memory_ms": 142.7
   },
   "all four": {
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
    "memory_ms": 1020.9
   },
   "all four, no bare surnames": {
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
    "memory_ms": 1054.6
   }
  }
 },
 "realistic": {
  "040": {
   "model_seconds": 3.541,
   "documents": 10,
   "words": 37985,
   "variants": {
    "off": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 2,
       "fn": 0,
       "precision": 0.9985,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 341,
       "fn": 48,
       "precision": 0.0833,
       "recall": 0.3924
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
      "boundary": 344,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 115.4
    },
    "labels": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 2,
       "fn": 0,
       "precision": 0.9985,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 341,
       "fn": 48,
       "precision": 0.0833,
       "recall": 0.3924
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
      "boundary": 344,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 143.6
    },
    "labels+pool": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 2,
       "fn": 0,
       "precision": 0.9985,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 343,
       "fn": 48,
       "precision": 0.0829,
       "recall": 0.3924
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
      "boundary": 346,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:pooled boundary": 2
     },
     "memory_ms": 126.7
    },
    "labels+pool+spread": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 2,
       "fn": 0,
       "precision": 0.9985,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 343,
       "fn": 48,
       "precision": 0.0829,
       "recall": 0.3924
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
      "boundary": 346,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:pooled boundary": 2
     },
     "memory_ms": 184.5
    },
    "all four": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 4,
       "fn": 0,
       "precision": 0.9971,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 343,
       "fn": 48,
       "precision": 0.0829,
       "recall": 0.3924
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
      "boundary": 348,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:alias boundary": 2,
      "memory:pooled boundary": 2
     },
     "memory_ms": 1165.7
    },
    "all four, no bare surnames": {
     "per_label": {
      "Person": {
       "tp": 1358,
       "fp": 2,
       "fn": 0,
       "precision": 0.9985,
       "recall": 1.0
      },
      "Organization": {
       "tp": 399,
       "fp": 1,
       "fn": 0,
       "precision": 0.9975,
       "recall": 1.0
      },
      "Street": {
       "tp": 31,
       "fp": 343,
       "fn": 48,
       "precision": 0.0829,
       "recall": 0.3924
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
      "boundary": 346,
      "missed": 48,
      "wrong label": 1
     },
     "controls_flagged": 0,
     "memory_added": {
      "memory:pooled boundary": 2
     },
     "memory_ms": 1140.6
    }
   }
  },
  "020": {
   "model_seconds": 1.678,
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
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 60.4
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
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 37.0
    },
    "labels+pool": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 38.1
    },
    "labels+pool+spread": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 68.2
    },
    "all four": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 619.3
    },
    "all four, no bare surnames": {
     "per_label": {
      "Person": {
       "tp": 680,
       "fp": 0,
       "fn": 0,
       "precision": 1.0,
       "recall": 1.0
      },
      "Organization": {
       "tp": 199,
       "fp": 1,
       "fn": 0,
       "precision": 0.995,
       "recall": 1.0
      },
      "Street": {
       "tp": 14,
       "fp": 174,
       "fn": 24,
       "precision": 0.0745,
       "recall": 0.3684
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
      "boundary": 174,
      "missed": 24,
      "invented": 1
     },
     "controls_flagged": 0,
     "memory_added": {},
     "memory_ms": 602.0
    }
   }
  }
 }
}
