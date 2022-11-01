window.BENCHMARK_DATA = {
  "lastUpdate": 1667311560795,
  "repoUrl": "https://github.com/huggingface/safetensors",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "huggingface",
            "username": "huggingface"
          },
          "committer": {
            "name": "huggingface",
            "username": "huggingface"
          },
          "id": "08b4ffb2bf3b05fc32d4a79abd140dca66e2e932",
          "message": "Splitting up tests from benchmarks.",
          "timestamp": "2022-11-01T07:54:34Z",
          "url": "https://github.com/huggingface/safetensors/pull/50/commits/08b4ffb2bf3b05fc32d4a79abd140dca66e2e932"
        },
        "date": 1667310830383,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1761837679361133,
            "unit": "iter/sec",
            "range": "stddev: 0.06050114010900778",
            "extra": "mean: 850.2072782000141 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1253939656762553,
            "unit": "iter/sec",
            "range": "stddev: 0.06444961940990351",
            "extra": "mean: 319.95966299999736 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.4366580735436916,
            "unit": "iter/sec",
            "range": "stddev: 0.011029530601841198",
            "extra": "mean: 290.9803590000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 3.011720158388529,
            "unit": "iter/sec",
            "range": "stddev: 0.010238305987880602",
            "extra": "mean: 332.0361612000056 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4736011081103566,
            "unit": "iter/sec",
            "range": "stddev: 0.10068621348608785",
            "extra": "mean: 678.6096959999782 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.8846619885488627,
            "unit": "iter/sec",
            "range": "stddev: 0.010457356083472449",
            "extra": "mean: 346.6610659999901 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "huggingface",
            "username": "huggingface"
          },
          "committer": {
            "name": "huggingface",
            "username": "huggingface"
          },
          "id": "4c5a241be09d14c808770f85627ab70127a32063",
          "message": "Splitting up tests from benchmarks.",
          "timestamp": "2022-11-01T07:54:34Z",
          "url": "https://github.com/huggingface/safetensors/pull/50/commits/4c5a241be09d14c808770f85627ab70127a32063"
        },
        "date": 1667311560266,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4807334447792253,
            "unit": "iter/sec",
            "range": "stddev: 0.042547544004970704",
            "extra": "mean: 675.3409963999957 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.240013874828148,
            "unit": "iter/sec",
            "range": "stddev: 0.10431122389692941",
            "extra": "mean: 308.6406535999913 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.539606906137727,
            "unit": "iter/sec",
            "range": "stddev: 0.009123308719396818",
            "extra": "mean: 282.5172473999828 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.6129205656866854,
            "unit": "iter/sec",
            "range": "stddev: 0.010663307385052927",
            "extra": "mean: 382.713509600012 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.9082718331257809,
            "unit": "iter/sec",
            "range": "stddev: 0.08112930679265079",
            "extra": "mean: 524.0343553999764 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.699145247781729,
            "unit": "iter/sec",
            "range": "stddev: 0.018136643384694662",
            "extra": "mean: 175.46490859997448 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}