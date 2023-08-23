window.BENCHMARK_DATA = {
  "lastUpdate": 1692801201988,
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
          "id": "15168a809b69b2297b919674ba61e113deb0dec1",
          "message": "Make GPU loading faster by removing all extra CPU copies.",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/33/commits/15168a809b69b2297b919674ba61e113deb0dec1"
        },
        "date": 1667336160596,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.9234793964248041,
            "unit": "iter/sec",
            "range": "stddev: 0.07689400323299302",
            "extra": "mean: 1.0828611919999958 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 1.9663002818321826,
            "unit": "iter/sec",
            "range": "stddev: 0.11837197429890743",
            "extra": "mean: 508.5693213999889 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.0818910599288447,
            "unit": "iter/sec",
            "range": "stddev: 0.008944479841901432",
            "extra": "mean: 480.3325299999983 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 1.600741779642756,
            "unit": "iter/sec",
            "range": "stddev: 0.01073290347342444",
            "extra": "mean: 624.7103766000123 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.0993446095506627,
            "unit": "iter/sec",
            "range": "stddev: 0.10641603073651551",
            "extra": "mean: 909.63287699999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 1.8989976417442387,
            "unit": "iter/sec",
            "range": "stddev: 0.0938069992411799",
            "extra": "mean: 526.5935975999923 msec\nrounds: 5"
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
          "id": "ef4b5376e576d4aacf4dc6f61bbb722fb7cc87ca",
          "message": "Adding M1 (mps) benchmarks (going first through CPU seems faster though? but PT is just as slow going directly on MPS.).",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/51/commits/ef4b5376e576d4aacf4dc6f61bbb722fb7cc87ca"
        },
        "date": 1667340479214,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2724705472134157,
            "unit": "iter/sec",
            "range": "stddev: 0.08044266918841757",
            "extra": "mean: 785.8728064000388 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.8580227329207966,
            "unit": "iter/sec",
            "range": "stddev: 0.08729102818479534",
            "extra": "mean: 349.89224840001043 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.0195338390758084,
            "unit": "iter/sec",
            "range": "stddev: 0.009859319389670097",
            "extra": "mean: 331.1769475999881 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.3880156878685623,
            "unit": "iter/sec",
            "range": "stddev: 0.010386881288471457",
            "extra": "mean: 418.75771799998347 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.210452945917001,
            "unit": "iter/sec",
            "range": "stddev: 0.055273436198714845",
            "extra": "mean: 826.1370286000101 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.1099034460376394,
            "unit": "iter/sec",
            "range": "stddev: 0.03661509552614141",
            "extra": "mean: 321.55339140001615 msec\nrounds: 5"
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
          "id": "87767a20a9aced71485d568047683f9499f782ff",
          "message": "Make GPU loading faster by removing all extra CPU copies.",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/33/commits/87767a20a9aced71485d568047683f9499f782ff"
        },
        "date": 1667341158680,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.414224537231388,
            "unit": "iter/sec",
            "range": "stddev: 0.05067191022293383",
            "extra": "mean: 707.1012937999853 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.96337995143045,
            "unit": "iter/sec",
            "range": "stddev: 0.0659099397657412",
            "extra": "mean: 252.30990019997535 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.634766916857968,
            "unit": "iter/sec",
            "range": "stddev: 0.01873333971286032",
            "extra": "mean: 215.76058040000134 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 5.594454318581948,
            "unit": "iter/sec",
            "range": "stddev: 0.010468423321213905",
            "extra": "mean: 178.74844319999283 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.162763433777928,
            "unit": "iter/sec",
            "range": "stddev: 0.02777295766742694",
            "extra": "mean: 462.37142000001086 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.9891785425878545,
            "unit": "iter/sec",
            "range": "stddev: 0.008593573038468876",
            "extra": "mean: 166.96780583333748 msec\nrounds: 6"
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
          "id": "23306433fdd497caf536c116f71044079a2a9226",
          "message": "[WIP] Re-enabling manylinux2014 builds.",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/52/commits/23306433fdd497caf536c116f71044079a2a9226"
        },
        "date": 1667374336783,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3993065739631083,
            "unit": "iter/sec",
            "range": "stddev: 0.08601422921345907",
            "extra": "mean: 714.639678399999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0423073852219016,
            "unit": "iter/sec",
            "range": "stddev: 0.07598818182350783",
            "extra": "mean: 328.69788399999607 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.3408914677369195,
            "unit": "iter/sec",
            "range": "stddev: 0.010191868864880103",
            "extra": "mean: 299.32130680000455 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 4.197308167047888,
            "unit": "iter/sec",
            "range": "stddev: 0.010022613189764641",
            "extra": "mean: 238.24793420001242 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3858421902849614,
            "unit": "iter/sec",
            "range": "stddev: 0.09543105227221804",
            "extra": "mean: 721.5828807999969 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.8600184713155032,
            "unit": "iter/sec",
            "range": "stddev: 0.016094418135405443",
            "extra": "mean: 259.0661178000005 msec\nrounds: 5"
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
          "id": "e4452e858bda08547ab50aacd62f1453c7561fde",
          "message": "[WIP] Re-enabling manylinux2014 builds.",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/52/commits/e4452e858bda08547ab50aacd62f1453c7561fde"
        },
        "date": 1667379906142,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 5.3793626218447494,
            "unit": "iter/sec",
            "range": "stddev: 0.00592131510509175",
            "extra": "mean: 185.89562933332596 msec\nrounds: 6"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 9.407259826160756,
            "unit": "iter/sec",
            "range": "stddev: 0.017415201010710195",
            "extra": "mean: 106.30088022221823 msec\nrounds: 9"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 9.477643605883697,
            "unit": "iter/sec",
            "range": "stddev: 0.003953560958778612",
            "extra": "mean: 105.51145850000125 msec\nrounds: 10"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 9.678783860419665,
            "unit": "iter/sec",
            "range": "stddev: 0.004687622906261461",
            "extra": "mean: 103.31876550001198 msec\nrounds: 10"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 4.196763934085295,
            "unit": "iter/sec",
            "range": "stddev: 0.029043732993601204",
            "extra": "mean: 238.2788299999902 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 10.140201478975584,
            "unit": "iter/sec",
            "range": "stddev: 0.009961507086089309",
            "extra": "mean: 98.61736988888954 msec\nrounds: 9"
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
          "id": "20adb08fcee2936e8700b07e5d939b1da6cc9a3d",
          "message": "Adding more docs:",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/53/commits/20adb08fcee2936e8700b07e5d939b1da6cc9a3d"
        },
        "date": 1667379962947,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.5403925390009043,
            "unit": "iter/sec",
            "range": "stddev: 0.03965716511039398",
            "extra": "mean: 649.1851750000023 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.113830609810627,
            "unit": "iter/sec",
            "range": "stddev: 0.11455916664830022",
            "extra": "mean: 321.1478482000075 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.342211803125016,
            "unit": "iter/sec",
            "range": "stddev: 0.009506538347663985",
            "extra": "mean: 299.20306039999787 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.445945543944888,
            "unit": "iter/sec",
            "range": "stddev: 0.009896208535684037",
            "extra": "mean: 408.83984619999865 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.2938161334286082,
            "unit": "iter/sec",
            "range": "stddev: 0.07801634596401112",
            "extra": "mean: 435.95473300001686 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.928781760080088,
            "unit": "iter/sec",
            "range": "stddev: 0.007198198089719236",
            "extra": "mean: 168.66871483333057 msec\nrounds: 6"
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
          "id": "b1a3929b989dd266c9aa4e9014db1415a6447b0f",
          "message": "Adding docs",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/55/commits/b1a3929b989dd266c9aa4e9014db1415a6447b0f"
        },
        "date": 1667382770535,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0995342815455929,
            "unit": "iter/sec",
            "range": "stddev: 0.03303437392237581",
            "extra": "mean: 909.4759633999956 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.8974786054828523,
            "unit": "iter/sec",
            "range": "stddev: 0.09032870302649294",
            "extra": "mean: 345.1276562000203 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.3045559558715576,
            "unit": "iter/sec",
            "range": "stddev: 0.010011286799984103",
            "extra": "mean: 302.6125183999966 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.961154896698312,
            "unit": "iter/sec",
            "range": "stddev: 0.012419020331433292",
            "extra": "mean: 337.7060758000198 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7575910518348794,
            "unit": "iter/sec",
            "range": "stddev: 0.02758585088863463",
            "extra": "mean: 568.9605661999849 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6339860263439534,
            "unit": "iter/sec",
            "range": "stddev: 0.05500283249453685",
            "extra": "mean: 275.179924400004 msec\nrounds: 5"
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
          "id": "d4ff161df4f619ec4f5693ab3a305e8447f73774",
          "message": "Adding docs",
          "timestamp": "2022-11-01T16:19:06Z",
          "url": "https://github.com/huggingface/safetensors/pull/55/commits/d4ff161df4f619ec4f5693ab3a305e8447f73774"
        },
        "date": 1667403069142,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4734789001331186,
            "unit": "iter/sec",
            "range": "stddev: 0.027445078580132364",
            "extra": "mean: 678.6659788000065 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1225284913769347,
            "unit": "iter/sec",
            "range": "stddev: 0.11317590919820404",
            "extra": "mean: 320.25328280000167 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.390581383179492,
            "unit": "iter/sec",
            "range": "stddev: 0.010055167244534192",
            "extra": "mean: 294.9346696000134 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 4.782799189969209,
            "unit": "iter/sec",
            "range": "stddev: 0.00971862062247003",
            "extra": "mean: 209.08258119999346 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7799574551364918,
            "unit": "iter/sec",
            "range": "stddev: 0.07889507295228787",
            "extra": "mean: 561.8111810000073 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.540831714575038,
            "unit": "iter/sec",
            "range": "stddev: 0.021990522046467755",
            "extra": "mean: 220.22397280001087 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c74743cf1ea2a706acef35b5f743664ed2b88f64",
          "message": "Benching only after merging. (#54)",
          "timestamp": "2022-11-02T16:52:53+01:00",
          "tree_id": "c24db81803903c08d2e348f0ec4df8898c84834a",
          "url": "https://github.com/huggingface/safetensors/commit/c74743cf1ea2a706acef35b5f743664ed2b88f64"
        },
        "date": 1667404735502,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1947676259609223,
            "unit": "iter/sec",
            "range": "stddev: 0.04325297884001762",
            "extra": "mean: 836.9828394000251 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.848410486741267,
            "unit": "iter/sec",
            "range": "stddev: 0.09530253830763538",
            "extra": "mean: 351.0729948000062 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.0329889386927342,
            "unit": "iter/sec",
            "range": "stddev: 0.010107059744606612",
            "extra": "mean: 329.70776360002674 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.3523641343529134,
            "unit": "iter/sec",
            "range": "stddev: 0.010251724044664609",
            "extra": "mean: 425.10425380000925 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6251960498779234,
            "unit": "iter/sec",
            "range": "stddev: 0.08129584768075743",
            "extra": "mean: 615.310380599999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.3549433802748934,
            "unit": "iter/sec",
            "range": "stddev: 0.010443379063565022",
            "extra": "mean: 298.0676233999702 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be0683f561f1f742c9253d46325945f6251a0770",
          "message": "Make GPU loading faster by removing all extra CPU copies. (#33)\n\n* Faster gpu load.\r\n\r\nUse cuda-sys directly.\r\n\r\nFor slice too.\r\n\r\nFun unsafe.\r\n\r\nReduce unsafe.\r\n\r\nRemoving CPU unsafe.\r\n\r\nUsing shared `cuda-sys` (temporary, we need to use torch's cuda version).\r\n\r\nTmp rework\r\n\r\nCleaner device.\r\n\r\nAdding some statistics...\r\n\r\nWarmup steps.\r\n\r\nRemoving unsafe GPU accesses.\r\n\r\nRemoving dead code.\r\n\r\nRemoving libloading.\r\n\r\nRevert \"Removing unsafe GPU accesses.\"\r\n\r\nThis reverts commit 5325ba2b73fffc16416130193da1690353e0a7db.\r\n\r\nUnsafe comments.\r\n\r\nUsing GILOnceCell for module reuse.\r\n\r\nFinding the lib through the real python workload.\r\nStill requires transitive library parsing.\r\n\r\nStable with global lib ref.\r\n\r\nAbort.\r\n\r\nWeird bug on torch 1.13.\r\n\r\nWe **need** to get the data_ptr within the loop ?\r\n\r\n* Fixing speedup by loading directly the loaded C library within Python.\r\n\r\n* Cleanup.\r\n\r\n* Finishing touches.\r\n\r\n* Remove some code duplication.\r\n\r\nSome very weird errors in calling the `cudaMemcpy` that fail depending\r\non order. Even aliasing the `Symbol` seem to be unsafe.\r\n\r\n* Adding some comments.\r\n\r\n* Better signature. Static symbol instead of lib.",
          "timestamp": "2022-11-02T16:53:50+01:00",
          "tree_id": "07902b31e4b3c421260a8c271ca6130efb8498e5",
          "url": "https://github.com/huggingface/safetensors/commit/be0683f561f1f742c9253d46325945f6251a0770"
        },
        "date": 1667404775663,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4376614890107318,
            "unit": "iter/sec",
            "range": "stddev: 0.05975114580496285",
            "extra": "mean: 695.574033000014 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1667596518081544,
            "unit": "iter/sec",
            "range": "stddev: 0.106637509904798",
            "extra": "mean: 315.78020119999337 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.445499483337568,
            "unit": "iter/sec",
            "range": "stddev: 0.010133627222081592",
            "extra": "mean: 290.23368160001155 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 5.2753430264375085,
            "unit": "iter/sec",
            "range": "stddev: 0.009863524780532144",
            "extra": "mean: 189.5611327999859 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6386920977630888,
            "unit": "iter/sec",
            "range": "stddev: 0.05051471643577982",
            "extra": "mean: 610.2427670000111 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.640529956555499,
            "unit": "iter/sec",
            "range": "stddev: 0.014328635652532755",
            "extra": "mean: 177.2883058333529 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4aec9801561e168b6d71b55168de202b4114a3a",
          "message": "Adding M1 (mps) benchmarks (going first through CPU seems faster though? (#51)\n\nbut PT is just as slow going directly on MPS.).",
          "timestamp": "2022-11-02T16:53:16+01:00",
          "tree_id": "e69f3b5138256aad4da0a617e562bc711dda4f58",
          "url": "https://github.com/huggingface/safetensors/commit/d4aec9801561e168b6d71b55168de202b4114a3a"
        },
        "date": 1667404790868,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1496411334041343,
            "unit": "iter/sec",
            "range": "stddev: 0.09651667950452923",
            "extra": "mean: 869.8366567999869 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.5438040824448733,
            "unit": "iter/sec",
            "range": "stddev: 0.10067693368220738",
            "extra": "mean: 393.1120352000107 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.9777006560718733,
            "unit": "iter/sec",
            "range": "stddev: 0.011834415037646574",
            "extra": "mean: 335.8295932000033 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.3075123479993396,
            "unit": "iter/sec",
            "range": "stddev: 0.01166423561954808",
            "extra": "mean: 433.36712839999336 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2201703785347369,
            "unit": "iter/sec",
            "range": "stddev: 0.048398102959783744",
            "extra": "mean: 819.5576761999973 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.090951687385472,
            "unit": "iter/sec",
            "range": "stddev: 0.011931145474421692",
            "extra": "mean: 244.44189920001236 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a76f9e82979bdbc9ba025edec47fa6005c633833",
          "message": "Adding more docs: (#53)\n\n- Simple python example on the README.md\r\n- Added `npz` and `arrow` formats to the table (not exhaustively)\r\n- Added `benefits` section.",
          "timestamp": "2022-11-02T16:53:33+01:00",
          "tree_id": "eada12be266a6c2a719be0a406c504b56f2f84bf",
          "url": "https://github.com/huggingface/safetensors/commit/a76f9e82979bdbc9ba025edec47fa6005c633833"
        },
        "date": 1667404800069,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.200630864476058,
            "unit": "iter/sec",
            "range": "stddev: 0.08736540124322084",
            "extra": "mean: 832.8954631999977 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.557507146633963,
            "unit": "iter/sec",
            "range": "stddev: 0.1018294582547695",
            "extra": "mean: 391.0057500000107 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.6490493188759108,
            "unit": "iter/sec",
            "range": "stddev: 0.019127147462848382",
            "extra": "mean: 377.4939156000073 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 3.732107474364879,
            "unit": "iter/sec",
            "range": "stddev: 0.011289711356490989",
            "extra": "mean: 267.9451239999935 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1204952533801815,
            "unit": "iter/sec",
            "range": "stddev: 0.1300142001792",
            "extra": "mean: 892.4625044000095 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.236833629597236,
            "unit": "iter/sec",
            "range": "stddev: 0.029853898745044065",
            "extra": "mean: 308.9438984000026 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f00cffa729be2fcfb869c17a697c27fc23b1f402",
          "message": "Supporting `safe_open` for TF and Flax. (#58)\n\n* Supporting `safe_open` for TF and Flax.\r\n\r\n* Move tensors only on Pytorch.",
          "timestamp": "2022-11-03T17:04:27+01:00",
          "tree_id": "93c9c44796ca4b211f32003acf4e0d438f48a2ca",
          "url": "https://github.com/huggingface/safetensors/commit/f00cffa729be2fcfb869c17a697c27fc23b1f402"
        },
        "date": 1667491959723,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1414805449763463,
            "unit": "iter/sec",
            "range": "stddev: 0.036337031617267646",
            "extra": "mean: 876.0552287999985 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.545628441128624,
            "unit": "iter/sec",
            "range": "stddev: 0.1038418596021533",
            "extra": "mean: 392.83030620000545 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.757274106959102,
            "unit": "iter/sec",
            "range": "stddev: 0.00967912121941567",
            "extra": "mean: 362.67703579999306 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 2.0984427735690474,
            "unit": "iter/sec",
            "range": "stddev: 0.009861086281262066",
            "extra": "mean: 476.54385079998747 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2701282204345614,
            "unit": "iter/sec",
            "range": "stddev: 0.14530792772616472",
            "extra": "mean: 787.3220859999947 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.880937067880167,
            "unit": "iter/sec",
            "range": "stddev: 0.010005243046268762",
            "extra": "mean: 347.1092829999975 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2f1da8be6258f520a7a849d68a56696ddb3464dc",
          "message": "Found out about `torch.asarray` + `torch.Storage` combo. (#56)\n\n* Found out about `torch.asarray` + `torch.Storage` combo.\r\n\r\nExtremely fast (I'm guessing it's just pointer arithmetic and actual\r\nload time would it when actually using the tensors.)\r\n\r\n* Dropping 1.10 support (no `asarray`.)\r\n\r\n* Accepting torch==1.10.0 in slow mode.\r\n\r\n* Adding a comment about `Storage`.",
          "timestamp": "2022-11-04T15:02:30+01:00",
          "tree_id": "b2a790aaee1d6e3f9261ad0e00ae454314aae4a4",
          "url": "https://github.com/huggingface/safetensors/commit/2f1da8be6258f520a7a849d68a56696ddb3464dc"
        },
        "date": 1667570893022,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2679113162564155,
            "unit": "iter/sec",
            "range": "stddev: 0.04949065266930508",
            "extra": "mean: 788.6986945999979 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.318022938062026,
            "unit": "iter/sec",
            "range": "stddev: 0.0722661863232775",
            "extra": "mean: 301.3842937999925 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.8714826161530174,
            "unit": "iter/sec",
            "range": "stddev: 0.010536996550223074",
            "extra": "mean: 258.29897720002464 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 190.58565561757982,
            "unit": "iter/sec",
            "range": "stddev: 0.0006219523492831864",
            "extra": "mean: 5.246984599966709 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6048717254997897,
            "unit": "iter/sec",
            "range": "stddev: 0.07911022158218482",
            "extra": "mean: 623.102758999994 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.012037754064634,
            "unit": "iter/sec",
            "range": "stddev: 0.013451905122312965",
            "extra": "mean: 332.00115059996733 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "119473908008378722ca3b59cf1f81cf7347a30e",
          "message": "Preparing for release. (#60)",
          "timestamp": "2022-11-04T15:42:43+01:00",
          "tree_id": "a71d4e1fd7677b07be29a9d72410d012cfe9ec5d",
          "url": "https://github.com/huggingface/safetensors/commit/119473908008378722ca3b59cf1f81cf7347a30e"
        },
        "date": 1667573351009,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.145381714692218,
            "unit": "iter/sec",
            "range": "stddev: 0.0689197106283898",
            "extra": "mean: 873.0713849999916 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.592060187654152,
            "unit": "iter/sec",
            "range": "stddev: 0.10759316500152569",
            "extra": "mean: 385.79351079999924 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.436084214063815,
            "unit": "iter/sec",
            "range": "stddev: 0.029641671081626148",
            "extra": "mean: 183.95594340000798 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 194.68430849148677,
            "unit": "iter/sec",
            "range": "stddev: 0.00011969847012696986",
            "extra": "mean: 5.136520799999289 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3133527664030828,
            "unit": "iter/sec",
            "range": "stddev: 0.05339020179810053",
            "extra": "mean: 761.4100533999931 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.047859642059664,
            "unit": "iter/sec",
            "range": "stddev: 0.051299599292381065",
            "extra": "mean: 328.0990982000162 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "764ff0faf4b36b1a0b42a2bd57869f3b8ccb2aaf",
          "message": "Adding the convert scripts which will now prevent converting models (#59)\n\n* Adding the convert scripts which will now prevent converting models\r\n\r\nin case they will trigger warnigns in the `transformers` side.\r\nEven if the model is perfectly fine, core maintainers fear an influx\r\nof opened issues.\r\n\r\nThis is perfectly legit.\r\nOn the `transformers` side fixes are on the way: https://github.com/huggingface/transformers/pull/20042\r\n\r\nWe can wait for this PR to hit `main` before communicating super widely.\r\n\r\nIn the meantime this script of convertion will now prevent converting\r\nmodels that would trigger such warnings (so the output of the script\r\n**will** depend on the `transformers` freshness.\r\n\r\n* Adding a nicer diff for the error when reloading.",
          "timestamp": "2022-11-04T17:38:58+01:00",
          "tree_id": "fcae620c33c26f5246f757ecd039a2db34add15e",
          "url": "https://github.com/huggingface/safetensors/commit/764ff0faf4b36b1a0b42a2bd57869f3b8ccb2aaf"
        },
        "date": 1667580293041,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4783809617931654,
            "unit": "iter/sec",
            "range": "stddev: 0.05922416427706766",
            "extra": "mean: 676.4156370000023 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.101111456077411,
            "unit": "iter/sec",
            "range": "stddev: 0.05815818075012042",
            "extra": "mean: 243.8363382000034 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.218556539955357,
            "unit": "iter/sec",
            "range": "stddev: 0.02408313574336143",
            "extra": "mean: 191.62386999998944 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 227.2820694510784,
            "unit": "iter/sec",
            "range": "stddev: 0.00015381513366589567",
            "extra": "mean: 4.399819142861361 msec\nrounds: 7"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8810317299829102,
            "unit": "iter/sec",
            "range": "stddev: 0.06028957922813192",
            "extra": "mean: 531.623142799981 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.936693692795961,
            "unit": "iter/sec",
            "range": "stddev: 0.020659453048524873",
            "extra": "mean: 168.4439271666444 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77566b96c68a100e8beccc92443bb3e1cbd6bb7c",
          "message": "Adding cargo audit check. (#64)\n\n* Adding cargo audit check.\r\n\r\n* Update command line.\r\n\r\n* Update affected crates.",
          "timestamp": "2022-11-08T15:47:22+01:00",
          "tree_id": "116690acc247d3c45ca286f3eefd229f81a67980",
          "url": "https://github.com/huggingface/safetensors/commit/77566b96c68a100e8beccc92443bb3e1cbd6bb7c"
        },
        "date": 1667919196046,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.097422063871392,
            "unit": "iter/sec",
            "range": "stddev: 0.04136285549705616",
            "extra": "mean: 911.2264395999887 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.5033585634258015,
            "unit": "iter/sec",
            "range": "stddev: 0.10981381540275884",
            "extra": "mean: 399.463350800022 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.548101954255351,
            "unit": "iter/sec",
            "range": "stddev: 0.03249410316775824",
            "extra": "mean: 219.87194000001864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 218.8057694879941,
            "unit": "iter/sec",
            "range": "stddev: 0.0002751499106134156",
            "extra": "mean: 4.57026340000084 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3910298391429805,
            "unit": "iter/sec",
            "range": "stddev: 0.13442657335608524",
            "extra": "mean: 718.8918396000076 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.323226241419488,
            "unit": "iter/sec",
            "range": "stddev: 0.01627326011666446",
            "extra": "mean: 300.91240479999897 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eeef55fe8f5022036f29609bd4c863730fc88df7",
          "message": "New version. (#63)\n\n* New version.\r\n\r\n* mmap2.",
          "timestamp": "2022-11-08T15:47:29+01:00",
          "tree_id": "48cb71306a651d8f145a7742f4974319e4911cc2",
          "url": "https://github.com/huggingface/safetensors/commit/eeef55fe8f5022036f29609bd4c863730fc88df7"
        },
        "date": 1667919211819,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.200909975699589,
            "unit": "iter/sec",
            "range": "stddev: 0.02903238635162233",
            "extra": "mean: 832.7018846000101 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.7527737571880264,
            "unit": "iter/sec",
            "range": "stddev: 0.0969971044751294",
            "extra": "mean: 363.26995540000553 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.5639104718907655,
            "unit": "iter/sec",
            "range": "stddev: 0.03196172102543622",
            "extra": "mean: 179.72970720001058 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 213.96434900415295,
            "unit": "iter/sec",
            "range": "stddev: 0.0003127323161647681",
            "extra": "mean: 4.673675799983812 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 0.9299454357993475,
            "unit": "iter/sec",
            "range": "stddev: 0.1194505037728026",
            "extra": "mean: 1.0753319082000075 sec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.9008914244986963,
            "unit": "iter/sec",
            "range": "stddev: 0.020145276627615755",
            "extra": "mean: 344.72162300000946 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "050d5d398bfc0142e893c6ba397e01fc05df4772",
          "message": "pip install .[dev] (#68)\n\n* Update setup.py\r\n\r\n* add click==8.0.4 to setup.py\r\n\r\n* gh actions use pip install .[dev]\r\n\r\n* Update binding local dev readme\r\n\r\n* fix gh action\r\n\r\n* fix gh action\r\n\r\n* fix gh action ?\r\n\r\n* don't install jax flax on windows\r\n\r\n* try removing unused deps",
          "timestamp": "2022-11-16T16:35:03+01:00",
          "tree_id": "0c282ad5b308864d61cec4860cc676acd6c3ccd0",
          "url": "https://github.com/huggingface/safetensors/commit/050d5d398bfc0142e893c6ba397e01fc05df4772"
        },
        "date": 1668613261534,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4073831684119906,
            "unit": "iter/sec",
            "range": "stddev: 0.01991015581292761",
            "extra": "mean: 710.5385529999921 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.917838556300919,
            "unit": "iter/sec",
            "range": "stddev: 0.0640741664632032",
            "extra": "mean: 255.24277879999318 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.466203924945665,
            "unit": "iter/sec",
            "range": "stddev: 0.02085664666656434",
            "extra": "mean: 223.903793199986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 218.54028464442,
            "unit": "iter/sec",
            "range": "stddev: 0.00038779852758449455",
            "extra": "mean: 4.575815400016836 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6919817101135997,
            "unit": "iter/sec",
            "range": "stddev: 0.07699478695831562",
            "extra": "mean: 591.022937199989 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.906046348909113,
            "unit": "iter/sec",
            "range": "stddev: 0.007917755371955378",
            "extra": "mean: 169.31800750001003 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "639791c6e2b79102e8c52b9cb32b6a724cad62db",
          "message": "Fix doc build actions (#70)\n\n* Fix doc build actions\r\n\r\n* Add remaining doc gh ymls",
          "timestamp": "2022-11-17T10:37:40+01:00",
          "tree_id": "97d4ccca7fe638179e074bc7f34c3af12b92f12d",
          "url": "https://github.com/huggingface/safetensors/commit/639791c6e2b79102e8c52b9cb32b6a724cad62db"
        },
        "date": 1668678211036,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3378876416490648,
            "unit": "iter/sec",
            "range": "stddev: 0.01618344512373872",
            "extra": "mean: 747.4469221999925 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0882434771454528,
            "unit": "iter/sec",
            "range": "stddev: 0.11911854375962681",
            "extra": "mean: 323.8086657999929 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.190950609597149,
            "unit": "iter/sec",
            "range": "stddev: 0.04380455930471843",
            "extra": "mean: 238.6093498000264 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 228.22728790395809,
            "unit": "iter/sec",
            "range": "stddev: 0.00017945243435985736",
            "extra": "mean: 4.381597000008242 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5818723673202721,
            "unit": "iter/sec",
            "range": "stddev: 0.0431793029626408",
            "extra": "mean: 632.1622531999992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.72955716288872,
            "unit": "iter/sec",
            "range": "stddev: 0.011355326839872006",
            "extra": "mean: 211.43628580000495 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d18e4b14c00afabe7cb8bc3cc96c744d02d9accb",
          "message": "Add installation section (#71)",
          "timestamp": "2022-11-17T11:16:11+01:00",
          "tree_id": "05958c07542cf6075e1c5f466680830cca8b2016",
          "url": "https://github.com/huggingface/safetensors/commit/d18e4b14c00afabe7cb8bc3cc96c744d02d9accb"
        },
        "date": 1668680521752,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2687437512356223,
            "unit": "iter/sec",
            "range": "stddev: 0.027686464571892663",
            "extra": "mean: 788.1812217999936 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.2940428944520956,
            "unit": "iter/sec",
            "range": "stddev: 0.06695592403349997",
            "extra": "mean: 303.57831759999954 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7136347111499823,
            "unit": "iter/sec",
            "range": "stddev: 0.041143548701278065",
            "extra": "mean: 269.27796559999706 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 214.98345229337048,
            "unit": "iter/sec",
            "range": "stddev: 0.00043107614288927996",
            "extra": "mean: 4.651520800007347 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.270474026243779,
            "unit": "iter/sec",
            "range": "stddev: 0.11747829659110559",
            "extra": "mean: 787.1077875999958 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.151527350690178,
            "unit": "iter/sec",
            "range": "stddev: 0.015857391170053955",
            "extra": "mean: 240.87520460000178 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "25b4d4794f46074eb73894d52e180052614dc47a",
          "message": "Fixing unimplemented into real exception. (#78)",
          "timestamp": "2022-11-17T20:08:25+01:00",
          "tree_id": "67cc68499a34e9445c986d7489270d9c6fc84591",
          "url": "https://github.com/huggingface/safetensors/commit/25b4d4794f46074eb73894d52e180052614dc47a"
        },
        "date": 1668712530069,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1593687278550722,
            "unit": "iter/sec",
            "range": "stddev: 0.029379748272018186",
            "extra": "mean: 862.5383589999728 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0030456823247507,
            "unit": "iter/sec",
            "range": "stddev: 0.07506717374466271",
            "extra": "mean: 332.99526739995144 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.887752051440298,
            "unit": "iter/sec",
            "range": "stddev: 0.039691374895293494",
            "extra": "mean: 257.2180496000328 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.64574372565627,
            "unit": "iter/sec",
            "range": "stddev: 0.00015567484798676882",
            "extra": "mean: 7.596143800014943 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3196060706573562,
            "unit": "iter/sec",
            "range": "stddev: 0.06542782950616803",
            "extra": "mean: 757.8019094000183 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.9478327942352975,
            "unit": "iter/sec",
            "range": "stddev: 0.025398930977120843",
            "extra": "mean: 253.3035343999927 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "KOLANICH@users.noreply.github.com",
            "name": "KOLANICH",
            "username": "KOLANICH"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d1373abcc44fb29bdb9db0dc45482d71244ad67c",
          "message": "Made the rust version consistent between `safetensors` and this crate. (#83)",
          "timestamp": "2022-11-18T10:19:54+01:00",
          "tree_id": "7b30fc3ae82f9543703154ca72044670e449a8ed",
          "url": "https://github.com/huggingface/safetensors/commit/d1373abcc44fb29bdb9db0dc45482d71244ad67c"
        },
        "date": 1668763549881,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2644879378822467,
            "unit": "iter/sec",
            "range": "stddev: 0.0740953147151298",
            "extra": "mean: 790.8339574000138 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.4058953038261603,
            "unit": "iter/sec",
            "range": "stddev: 0.055113454393563154",
            "extra": "mean: 293.6085553999874 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.288842162812033,
            "unit": "iter/sec",
            "range": "stddev: 0.04143626969543044",
            "extra": "mean: 233.1631619999598 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 217.57393243805083,
            "unit": "iter/sec",
            "range": "stddev: 0.00017898831471550888",
            "extra": "mean: 4.596138833335317 msec\nrounds: 6"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.365769867262804,
            "unit": "iter/sec",
            "range": "stddev: 0.07182144274594406",
            "extra": "mean: 732.1877747999679 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.9392648028926796,
            "unit": "iter/sec",
            "range": "stddev: 0.015096393356353374",
            "extra": "mean: 253.85447539999862 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2bb7ace99104da75fd7bc53465c81a5521d1900c",
          "message": "Preventing big endian files to be saved. (#82)",
          "timestamp": "2022-11-18T10:19:27+01:00",
          "tree_id": "3a1e09223917fcd820a0759c7cb6dc81df68d6d5",
          "url": "https://github.com/huggingface/safetensors/commit/2bb7ace99104da75fd7bc53465c81a5521d1900c"
        },
        "date": 1668763608142,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.18247217344461,
            "unit": "iter/sec",
            "range": "stddev: 0.03310017808516694",
            "extra": "mean: 845.6858625999985 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0311702003334706,
            "unit": "iter/sec",
            "range": "stddev: 0.05710093073326689",
            "extra": "mean: 329.90559219999795 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.208666373290835,
            "unit": "iter/sec",
            "range": "stddev: 0.04763653047459611",
            "extra": "mean: 191.98772360000476 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 117.78923736174092,
            "unit": "iter/sec",
            "range": "stddev: 0.0004948721392214658",
            "extra": "mean: 8.48974000000453 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2712817581521076,
            "unit": "iter/sec",
            "range": "stddev: 0.036294544221817646",
            "extra": "mean: 786.6076844000077 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.477179825847428,
            "unit": "iter/sec",
            "range": "stddev: 0.015400697914376081",
            "extra": "mean: 287.5893827999789 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f23f53accd104c2ad7abdfe716706c991356b474",
          "message": "Trigger doc build on specific changes (#87)\n\n* Trigger doc build on specific changes\r\n\r\n* chore",
          "timestamp": "2022-11-18T11:40:30+01:00",
          "tree_id": "c3dc7b5acefbad011b0c2d4a5edde88cdac54f46",
          "url": "https://github.com/huggingface/safetensors/commit/f23f53accd104c2ad7abdfe716706c991356b474"
        },
        "date": 1668768433002,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0923170426928144,
            "unit": "iter/sec",
            "range": "stddev: 0.028698350025099746",
            "extra": "mean: 915.4851210000061 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.7562296797965473,
            "unit": "iter/sec",
            "range": "stddev: 0.07538784225292906",
            "extra": "mean: 362.8144662000068 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.6207507720721406,
            "unit": "iter/sec",
            "range": "stddev: 0.05062066885847581",
            "extra": "mean: 276.18581419999373 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 89.48966442711134,
            "unit": "iter/sec",
            "range": "stddev: 0.00075854444715374",
            "extra": "mean: 11.174474799986456 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.397033230948958,
            "unit": "iter/sec",
            "range": "stddev: 0.08736163173476133",
            "extra": "mean: 715.802586399991 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.690570842898221,
            "unit": "iter/sec",
            "range": "stddev: 0.01565486707927653",
            "extra": "mean: 270.9607923999897 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "24647b6e6a9a5a275c38a9d7d7e6d93cd0ab444c",
          "message": "delete_doc_comment.yml should only run on pull request closed (#88)",
          "timestamp": "2022-11-18T12:04:01+01:00",
          "tree_id": "3e03cb41105bc1154c09fc3df36e56fe981416bd",
          "url": "https://github.com/huggingface/safetensors/commit/24647b6e6a9a5a275c38a9d7d7e6d93cd0ab444c"
        },
        "date": 1668769811283,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1512874532597261,
            "unit": "iter/sec",
            "range": "stddev: 0.018732029086328005",
            "extra": "mean: 868.5928064000223 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.2031569371762423,
            "unit": "iter/sec",
            "range": "stddev: 0.06600868696414217",
            "extra": "mean: 312.1920092000096 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.156019885750656,
            "unit": "iter/sec",
            "range": "stddev: 0.007519438366453126",
            "extra": "mean: 240.61482559999376 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 217.78456646136706,
            "unit": "iter/sec",
            "range": "stddev: 0.00023308615895864792",
            "extra": "mean: 4.5916936000026 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2948406150400427,
            "unit": "iter/sec",
            "range": "stddev: 0.025996838963325906",
            "extra": "mean: 772.2958242000118 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7747311219952575,
            "unit": "iter/sec",
            "range": "stddev: 0.012645247354191263",
            "extra": "mean: 264.9195314000053 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a3116ff4d3330a4184aeabae68d7d6651d72334d",
          "message": "[docs] Loading speed benchmark (#80)\n\n* [docs] Loading speed benchmark\r\n\r\n* Add speed reference in index\r\n\r\n* chore\r\n\r\n* make print statement clearer\r\n\r\n* remove repetition\r\n\r\n* add notebooks folder\r\n\r\n* fuse `device` lines\r\n\r\n* add specs\r\n\r\n* rm unneded lines\r\n\r\n* Update docs/source/speed.mdx\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2022-11-18T13:18:05+01:00",
          "tree_id": "14523d04c34d2de09a38e31ada7be843ae0b7a2a",
          "url": "https://github.com/huggingface/safetensors/commit/a3116ff4d3330a4184aeabae68d7d6651d72334d"
        },
        "date": 1668774240498,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.41854486690608,
            "unit": "iter/sec",
            "range": "stddev: 0.01642550421256956",
            "extra": "mean: 704.9477413999966 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.796249261541387,
            "unit": "iter/sec",
            "range": "stddev: 0.07590481895039233",
            "extra": "mean: 263.4178977999909 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.061773683276674,
            "unit": "iter/sec",
            "range": "stddev: 0.013216460415113086",
            "extra": "mean: 246.1978627999997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 228.1939045754306,
            "unit": "iter/sec",
            "range": "stddev: 0.0001836832437984925",
            "extra": "mean: 4.382238000005145 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.92548301183252,
            "unit": "iter/sec",
            "range": "stddev: 0.06450403824399255",
            "extra": "mean: 519.3502065999951 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.698562540852467,
            "unit": "iter/sec",
            "range": "stddev: 0.016058379136666454",
            "extra": "mean: 212.83105019999766 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03e3d0b63b332d0b34accc6c21730aa8991f2c2b",
          "message": "Add missing doc build token (#89)",
          "timestamp": "2022-11-18T13:47:01+01:00",
          "tree_id": "d0b0f9dd210d26ac2401888ff82942a83021a3dc",
          "url": "https://github.com/huggingface/safetensors/commit/03e3d0b63b332d0b34accc6c21730aa8991f2c2b"
        },
        "date": 1668775976420,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4394406805191224,
            "unit": "iter/sec",
            "range": "stddev: 0.014511063452552602",
            "extra": "mean: 694.7142828000096 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.021935329620826,
            "unit": "iter/sec",
            "range": "stddev: 0.06861879569588136",
            "extra": "mean: 248.63651899999013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.086617237823067,
            "unit": "iter/sec",
            "range": "stddev: 0.015370943470001917",
            "extra": "mean: 196.59430880000173 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 201.20029658550058,
            "unit": "iter/sec",
            "range": "stddev: 0.0006951950466933328",
            "extra": "mean: 4.970171599995865 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5950949981492286,
            "unit": "iter/sec",
            "range": "stddev: 0.10399397551604837",
            "extra": "mean: 626.9219082000063 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.515614965200495,
            "unit": "iter/sec",
            "range": "stddev: 0.005981824935314808",
            "extra": "mean: 181.3034459999964 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a8e324da16898d246e63c76873de0c32560bbec6",
          "message": "[docs] Fixes the colab btn on speed page (#91)",
          "timestamp": "2022-11-18T14:15:53+01:00",
          "tree_id": "2b5d56bbc1647a7f23c673e0a37b3a209cc8209c",
          "url": "https://github.com/huggingface/safetensors/commit/a8e324da16898d246e63c76873de0c32560bbec6"
        },
        "date": 1668777705689,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3969696669119924,
            "unit": "iter/sec",
            "range": "stddev: 0.019703343598809105",
            "extra": "mean: 715.8351563999986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.834063907181543,
            "unit": "iter/sec",
            "range": "stddev: 0.07464657925458416",
            "extra": "mean: 260.8198570000127 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.138466778129653,
            "unit": "iter/sec",
            "range": "stddev: 0.015215764782861851",
            "extra": "mean: 241.63538180000614 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 225.40624065198972,
            "unit": "iter/sec",
            "range": "stddev: 0.00018428177316964369",
            "extra": "mean: 4.436434399985956 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.878107024993401,
            "unit": "iter/sec",
            "range": "stddev: 0.09497847428051742",
            "extra": "mean: 532.4510193999799 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.009871578842393,
            "unit": "iter/sec",
            "range": "stddev: 0.015452046426182541",
            "extra": "mean: 249.38454519999596 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b8561611f61ce5b204ca5f5597f0e4d64f0b896f",
          "message": "[docs] fix notebook automatic creation (#92)",
          "timestamp": "2022-11-18T14:39:31+01:00",
          "tree_id": "a06c78ea637c1b205718c6a585ca96d6f8d60ae4",
          "url": "https://github.com/huggingface/safetensors/commit/b8561611f61ce5b204ca5f5597f0e4d64f0b896f"
        },
        "date": 1668779124203,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4885686937602876,
            "unit": "iter/sec",
            "range": "stddev: 0.016454980770729235",
            "extra": "mean: 671.78626299999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.035690031591905,
            "unit": "iter/sec",
            "range": "stddev: 0.06124525999599909",
            "extra": "mean: 247.78909980000208 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.886423735599611,
            "unit": "iter/sec",
            "range": "stddev: 0.0326005168371902",
            "extra": "mean: 169.8824353999953 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 234.76228800046798,
            "unit": "iter/sec",
            "range": "stddev: 0.000040113827964414417",
            "extra": "mean: 4.259627934781444 msec\nrounds: 138"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7797492191183248,
            "unit": "iter/sec",
            "range": "stddev: 0.05290534917351666",
            "extra": "mean: 561.8769146000204 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.313455952494839,
            "unit": "iter/sec",
            "range": "stddev: 0.03218822507810663",
            "extra": "mean: 188.2014283999979 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "11c21549cca3ec3f871b44a8804e2f9ce4df2ac6",
          "message": "Torch fixes (#90)\n\n* Fix `KeyError: torch.float64` when serializing tensors of doubles and `KeyError: BF16` when parsing.\r\n\r\n* Fix non-working `_view2torch` because of the forgotten returns.\r\nAdded some tests.\r\n\r\n* Test in the same style.\r\n\r\n* Like previous style.\r\n\r\nCo-authored-by: KOLANICH <kolan_n@mail.ru>",
          "timestamp": "2022-11-18T14:40:12+01:00",
          "tree_id": "20e576d6f23dbc64faa256d56c2f1a447c396fd5",
          "url": "https://github.com/huggingface/safetensors/commit/11c21549cca3ec3f871b44a8804e2f9ce4df2ac6"
        },
        "date": 1668779171733,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1553624239403764,
            "unit": "iter/sec",
            "range": "stddev: 0.02488906321445527",
            "extra": "mean: 865.529274000005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.557268212296707,
            "unit": "iter/sec",
            "range": "stddev: 0.05042642579917077",
            "extra": "mean: 281.1145914000008 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.801206173610491,
            "unit": "iter/sec",
            "range": "stddev: 0.04603616291867478",
            "extra": "mean: 208.28099520000478 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 240.94099215353313,
            "unit": "iter/sec",
            "range": "stddev: 0.0002992712080123191",
            "extra": "mean: 4.150393800000529 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.406624372065602,
            "unit": "iter/sec",
            "range": "stddev: 0.04780854881817957",
            "extra": "mean: 710.9218494000061 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.703873461826983,
            "unit": "iter/sec",
            "range": "stddev: 0.020248584184639484",
            "extra": "mean: 269.9876252000081 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "736d990c335e0e2efcfcc212731635a2d944aab3",
          "message": "Accepting `pathlib.Path` in `save_file/load_file/safe_open`. (#86)\n\n* Accepting `pathlib.Path` in `save_file/load_file/safe_open`.\r\n\r\n* as_path.\r\n\r\n* Update docstrings.\r\n\r\n- Removed unused `:obj:`\r\n- Put `os.PathLike` everywhere it's needed\r\n- python stub.py",
          "timestamp": "2022-11-18T15:24:35+01:00",
          "tree_id": "3eeb62d600a220c8119eb524dd011b7b0baf1721",
          "url": "https://github.com/huggingface/safetensors/commit/736d990c335e0e2efcfcc212731635a2d944aab3"
        },
        "date": 1668781836813,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3214734729356128,
            "unit": "iter/sec",
            "range": "stddev: 0.007090689232857639",
            "extra": "mean: 756.7310434000092 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.525303617249789,
            "unit": "iter/sec",
            "range": "stddev: 0.09140790113077457",
            "extra": "mean: 283.66351060001307 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.191119617866332,
            "unit": "iter/sec",
            "range": "stddev: 0.04002723111791185",
            "extra": "mean: 238.59972779996497 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 229.2965402281221,
            "unit": "iter/sec",
            "range": "stddev: 0.00018098517284721778",
            "extra": "mean: 4.361164799979633 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.157039190819336,
            "unit": "iter/sec",
            "range": "stddev: 0.0454336507431121",
            "extra": "mean: 463.5984381999833 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.8102553227894775,
            "unit": "iter/sec",
            "range": "stddev: 0.016627162392253352",
            "extra": "mean: 172.10947616668668 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "19379d4917422a13528505afd7eb2e290ac1134b",
          "message": "[docs] we want markdnwo headings to show (#94)",
          "timestamp": "2022-11-18T15:47:04+01:00",
          "tree_id": "a83b7624e5966fb803536bc9be0304f854ecc6c7",
          "url": "https://github.com/huggingface/safetensors/commit/19379d4917422a13528505afd7eb2e290ac1134b"
        },
        "date": 1668783179014,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3197967844696663,
            "unit": "iter/sec",
            "range": "stddev: 0.015290585833163964",
            "extra": "mean: 757.692405199964 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1251293959822553,
            "unit": "iter/sec",
            "range": "stddev: 0.11597195592833444",
            "extra": "mean: 319.98675040003945 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.016594834687422,
            "unit": "iter/sec",
            "range": "stddev: 0.030992474607123083",
            "extra": "mean: 248.96710800003348 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 230.72656765187216,
            "unit": "iter/sec",
            "range": "stddev: 0.00020174672380314265",
            "extra": "mean: 4.334134600003381 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7857516742522017,
            "unit": "iter/sec",
            "range": "stddev: 0.007036854939818391",
            "extra": "mean: 559.9882751999985 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.852282243392146,
            "unit": "iter/sec",
            "range": "stddev: 0.015990167932019936",
            "extra": "mean: 206.08858880000298 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4881035530f7e31b38a822da569e48aa91d3f06c",
          "message": "Ignore unsound criterion. (#100)",
          "timestamp": "2022-11-22T13:12:42+01:00",
          "tree_id": "16752fc86d82a306f45057fd85325f3710ad259b",
          "url": "https://github.com/huggingface/safetensors/commit/4881035530f7e31b38a822da569e48aa91d3f06c"
        },
        "date": 1669119626331,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0187636249154353,
            "unit": "iter/sec",
            "range": "stddev: 0.0500553198103045",
            "extra": "mean: 981.5819641999951 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.4561722176864307,
            "unit": "iter/sec",
            "range": "stddev: 0.0800705402823296",
            "extra": "mean: 407.1375748000037 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.1024170660209593,
            "unit": "iter/sec",
            "range": "stddev: 0.0419129612067538",
            "extra": "mean: 322.3293254000055 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 110.39909826022924,
            "unit": "iter/sec",
            "range": "stddev: 0.00147729981973821",
            "extra": "mean: 9.058044999994763 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.222879434506962,
            "unit": "iter/sec",
            "range": "stddev: 0.08841909067401119",
            "extra": "mean: 817.742102599982 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.925814820643907,
            "unit": "iter/sec",
            "range": "stddev: 0.030052409743994875",
            "extra": "mean: 341.7851303999896 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fb46d7a54be5bd9ef2419b1606ec4625951e2c3",
          "message": "Fixing Windows overflow issue with `c_long`. (#99)\n\n* Fixing Windows overflow issue with `c_long`.\r\n\r\n* Adding comment to remove the code once fixed.\r\n\r\n* Fixing just like PyO3.",
          "timestamp": "2022-11-22T15:07:31+01:00",
          "tree_id": "3d91346cd6b86e7913deb3483ea61de9b42d613d",
          "url": "https://github.com/huggingface/safetensors/commit/5fb46d7a54be5bd9ef2419b1606ec4625951e2c3"
        },
        "date": 1669126459825,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.9812078097113685,
            "unit": "iter/sec",
            "range": "stddev: 0.029014071648890936",
            "extra": "mean: 1.0191521002000172 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.469238445967857,
            "unit": "iter/sec",
            "range": "stddev: 0.0975401065530492",
            "extra": "mean: 404.9831645999802 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.1571060419037553,
            "unit": "iter/sec",
            "range": "stddev: 0.02680745565877583",
            "extra": "mean: 316.74577500000396 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 122.80460971172688,
            "unit": "iter/sec",
            "range": "stddev: 0.000864453086106275",
            "extra": "mean: 8.143016799999714 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1698394559498564,
            "unit": "iter/sec",
            "range": "stddev: 0.04032394515268611",
            "extra": "mean: 854.8181504000013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.9654184207941765,
            "unit": "iter/sec",
            "range": "stddev: 0.057620722928500456",
            "extra": "mean: 337.22053960000267 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "383871d475875c17c3489de505a4214e3685df2b",
          "message": "Adding suppot for 3.11 where possible (conda does not support 3.11 yet). (#97)\n\n* Adding suppot for 3.11 where possible (conda does not support 3.11 yet).\r\n\r\n* Update .github/workflows/python-release.yml\r\n\r\n* Update .github/workflows/python-release-conda.yml\r\n\r\nCo-authored-by: Sylvain Gugger <35901082+sgugger@users.noreply.github.com>\r\n\r\nCo-authored-by: Sylvain Gugger <35901082+sgugger@users.noreply.github.com>",
          "timestamp": "2022-11-22T15:36:11+01:00",
          "tree_id": "10203e8e8c72c785223df7cba6eca71d0ee49bbd",
          "url": "https://github.com/huggingface/safetensors/commit/383871d475875c17c3489de505a4214e3685df2b"
        },
        "date": 1669128131241,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3758689602766425,
            "unit": "iter/sec",
            "range": "stddev: 0.05126254843922128",
            "extra": "mean: 726.8134021999686 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.489830457461386,
            "unit": "iter/sec",
            "range": "stddev: 0.05388336405602394",
            "extra": "mean: 286.5468716000123 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.303560938979685,
            "unit": "iter/sec",
            "range": "stddev: 0.029994731573023847",
            "extra": "mean: 232.36571160000494 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 200.58047990981058,
            "unit": "iter/sec",
            "range": "stddev: 0.0008865505759917868",
            "extra": "mean: 4.985529999976279 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6732785095677132,
            "unit": "iter/sec",
            "range": "stddev: 0.05861653797188899",
            "extra": "mean: 597.6291420000052 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.407666530812864,
            "unit": "iter/sec",
            "range": "stddev: 0.014553135907535245",
            "extra": "mean: 293.4559444000115 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e5c91853aa14682fdde294efa5806fea88aca43c",
          "message": "Making convertion working with diffusers/stable-diffusion. (#96)\n\n* Making convertion working with diffusers/stable-diffusion.\r\n\r\n* Update bindings/python/convert.py\r\n\r\nCo-authored-by: Sylvain Gugger <35901082+sgugger@users.noreply.github.com>\r\n\r\nCo-authored-by: Sylvain Gugger <35901082+sgugger@users.noreply.github.com>",
          "timestamp": "2022-11-22T15:25:30+01:00",
          "tree_id": "8b5c621c6d3dfc584d1b341d27d3f2a04282b0d7",
          "url": "https://github.com/huggingface/safetensors/commit/e5c91853aa14682fdde294efa5806fea88aca43c"
        },
        "date": 1669128868097,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1862751180327613,
            "unit": "iter/sec",
            "range": "stddev: 0.0316307669166667",
            "extra": "mean: 842.9747743999997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.043250471189642,
            "unit": "iter/sec",
            "range": "stddev: 0.06973002987738959",
            "extra": "mean: 328.5960224000519 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.3669993719506004,
            "unit": "iter/sec",
            "range": "stddev: 0.011387554800112549",
            "extra": "mean: 297.0003524000276 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 121.67884290553769,
            "unit": "iter/sec",
            "range": "stddev: 0.0001904246925087757",
            "extra": "mean: 8.218355600047289 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3975348484789567,
            "unit": "iter/sec",
            "range": "stddev: 0.1475143610886606",
            "extra": "mean: 715.5456632000096 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.2194914979697207,
            "unit": "iter/sec",
            "range": "stddev: 0.011769740751905398",
            "extra": "mean: 310.60805739994066 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bba82fab0cceccc973e5461fc7d6651440221ff7",
          "message": "Small qol. (#101)",
          "timestamp": "2022-11-22T16:07:47+01:00",
          "tree_id": "57b3ee7d534b3c461b4c9349d004bdc8eeffc3e8",
          "url": "https://github.com/huggingface/safetensors/commit/bba82fab0cceccc973e5461fc7d6651440221ff7"
        },
        "date": 1669130044298,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1515087924229965,
            "unit": "iter/sec",
            "range": "stddev: 0.015199155605046748",
            "extra": "mean: 868.4258483999997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.261298459995918,
            "unit": "iter/sec",
            "range": "stddev: 0.06799299302382875",
            "extra": "mean: 306.6263367999909 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7082054266363342,
            "unit": "iter/sec",
            "range": "stddev: 0.026295248212588742",
            "extra": "mean: 269.6722227999885 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 216.87226301705743,
            "unit": "iter/sec",
            "range": "stddev: 0.00035367680297954167",
            "extra": "mean: 4.6110092000162695 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.524600624379271,
            "unit": "iter/sec",
            "range": "stddev: 0.10863755525494279",
            "extra": "mean: 655.9094781999988 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.5269608754440083,
            "unit": "iter/sec",
            "range": "stddev: 0.04980949662589909",
            "extra": "mean: 283.53022200001305 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b57fd895c366c631d75ac3717face487cb68be5b",
          "message": "convert hotfix. (#102)",
          "timestamp": "2022-11-22T17:00:05+01:00",
          "tree_id": "a9da5a2a52eb32971fa613b63268fb963b712483",
          "url": "https://github.com/huggingface/safetensors/commit/b57fd895c366c631d75ac3717face487cb68be5b"
        },
        "date": 1669133221383,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2440489351684492,
            "unit": "iter/sec",
            "range": "stddev: 0.033076607038164",
            "extra": "mean: 803.8269008000043 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.146624809273786,
            "unit": "iter/sec",
            "range": "stddev: 0.06538381284970891",
            "extra": "mean: 317.8008375999525 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.39344299533957,
            "unit": "iter/sec",
            "range": "stddev: 0.010632895981934431",
            "extra": "mean: 294.68595799999093 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 120.30804634654355,
            "unit": "iter/sec",
            "range": "stddev: 0.0001876516807584906",
            "extra": "mean: 8.311995999997634 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5213692044855873,
            "unit": "iter/sec",
            "range": "stddev: 0.0752272050254711",
            "extra": "mean: 657.3026435999964 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.077888249519469,
            "unit": "iter/sec",
            "range": "stddev: 0.014652133693861497",
            "extra": "mean: 245.22496419999698 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "27cb200d5c4871cd040da63051cedaa90b435ae4",
          "message": "Preparing for release. (#104)",
          "timestamp": "2022-11-23T11:47:35+01:00",
          "tree_id": "cfdf7dfe3e68c45f593d727c1d8808381e8d2762",
          "url": "https://github.com/huggingface/safetensors/commit/27cb200d5c4871cd040da63051cedaa90b435ae4"
        },
        "date": 1669200816286,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0760860015421962,
            "unit": "iter/sec",
            "range": "stddev: 0.01481998239083619",
            "extra": "mean: 929.2937540000025 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9737459331271436,
            "unit": "iter/sec",
            "range": "stddev: 0.0857424002321812",
            "extra": "mean: 336.276206000025 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.030174619151537,
            "unit": "iter/sec",
            "range": "stddev: 0.0221036310829433",
            "extra": "mean: 198.8002556000083 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 234.78776055102537,
            "unit": "iter/sec",
            "range": "stddev: 0.00019233870990562285",
            "extra": "mean: 4.259165800010578 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2585186855049286,
            "unit": "iter/sec",
            "range": "stddev: 0.09531432075430117",
            "extra": "mean: 794.5849446000011 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6750840925028276,
            "unit": "iter/sec",
            "range": "stddev: 0.0474458184437986",
            "extra": "mean: 272.10261719997106 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a4eb42bb1b98e41f442a94425075d358366e0ae",
          "message": "Improve error message when saving bogus object. (#106)",
          "timestamp": "2022-11-23T15:41:36+01:00",
          "tree_id": "ef537bfbb22c91c8abfe9f722a59623240f37b66",
          "url": "https://github.com/huggingface/safetensors/commit/0a4eb42bb1b98e41f442a94425075d358366e0ae"
        },
        "date": 1669214852574,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4320056506117744,
            "unit": "iter/sec",
            "range": "stddev: 0.021220209202596256",
            "extra": "mean: 698.3212668000192 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.034359337802358,
            "unit": "iter/sec",
            "range": "stddev: 0.06141639880853322",
            "extra": "mean: 247.87083060001578 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.894070429466336,
            "unit": "iter/sec",
            "range": "stddev: 0.04227248926999399",
            "extra": "mean: 169.66203780000342 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 211.71125700952376,
            "unit": "iter/sec",
            "range": "stddev: 0.0005223333394342848",
            "extra": "mean: 4.723414399995818 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.973509051211131,
            "unit": "iter/sec",
            "range": "stddev: 0.13659111175717165",
            "extra": "mean: 506.711636 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.590023510084652,
            "unit": "iter/sec",
            "range": "stddev: 0.012344362941798772",
            "extra": "mean: 178.89012420000654 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9b4a932d57a60dfdf4aa149c8fe7763c64172c4",
          "message": "Fix `convert.py` to handle diffusers, and its transformers part (#105)\n\nTurns out `diffusers` does rely on `transformers`:\r\n\r\n`pytorch_model.safetensors` was not a recognized filename by\r\n`transformers` which expects it to be named `model.safetensors`.\r\n\r\nThe convert_generic works by simply creating the new files and changing\r\nthe extension, which broke this.\r\n\r\nCurrent proposed fix is to handle specially files named\r\n`pytorch_model.bin` since they are more likely to be `transformers`\r\nfiles.\r\n\r\nThis is however not necessarily true, we might want to protect with\r\nchecking against `model_info.library_name` though I feel like this is\r\nbetter to just check against the name, since it's still likely to be a\r\ntransformers/diffusers/sentence-transformers thing.",
          "timestamp": "2022-11-23T15:41:15+01:00",
          "tree_id": "a95372257fa36e9b6ba7e86af6f6fbc921cb1f51",
          "url": "https://github.com/huggingface/safetensors/commit/f9b4a932d57a60dfdf4aa149c8fe7763c64172c4"
        },
        "date": 1669214880017,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0588198282509813,
            "unit": "iter/sec",
            "range": "stddev: 0.02613243032934004",
            "extra": "mean: 944.4477458000165 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9799010028018214,
            "unit": "iter/sec",
            "range": "stddev: 0.07806879916318188",
            "extra": "mean: 335.5816179999806 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.560214295649887,
            "unit": "iter/sec",
            "range": "stddev: 0.04155312754092277",
            "extra": "mean: 280.88196859999925 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 186.25091290943567,
            "unit": "iter/sec",
            "range": "stddev: 0.0002557451833081166",
            "extra": "mean: 5.369101199983106 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1210282627255728,
            "unit": "iter/sec",
            "range": "stddev: 0.07586994340087701",
            "extra": "mean: 892.0381700000007 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.953337133043895,
            "unit": "iter/sec",
            "range": "stddev: 0.018299152269250767",
            "extra": "mean: 252.95085299999297 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6930dfb5a79772a94b369e531e38af331b790bd4",
          "message": "Fixes https://github.com/huggingface/safetensors/actions/runs/3534003128/jobs/5930362952 (#107)",
          "timestamp": "2022-11-23T18:11:10+01:00",
          "tree_id": "f57a66908f2a946bbc97b3916770e291c82053a9",
          "url": "https://github.com/huggingface/safetensors/commit/6930dfb5a79772a94b369e531e38af331b790bd4"
        },
        "date": 1669223810445,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2841486580177794,
            "unit": "iter/sec",
            "range": "stddev: 0.009102563349220827",
            "extra": "mean: 778.7260406000087 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3383915841403446,
            "unit": "iter/sec",
            "range": "stddev: 0.0687752994920189",
            "extra": "mean: 299.5454471999892 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.62221117259446,
            "unit": "iter/sec",
            "range": "stddev: 0.01178925956736208",
            "extra": "mean: 276.07446180001034 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 220.83677084160877,
            "unit": "iter/sec",
            "range": "stddev: 0.00048162585369056443",
            "extra": "mean: 4.528231400001914 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6260593441252005,
            "unit": "iter/sec",
            "range": "stddev: 0.06997537704315247",
            "extra": "mean: 614.9837050000087 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.439104735061563,
            "unit": "iter/sec",
            "range": "stddev: 0.014206065659950374",
            "extra": "mean: 290.77334860000974 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3836e27dab84717dca64c0c7811eb6a4e4f07231",
          "message": "M1 fix (pyenv cannot use `3.11`. (#108)",
          "timestamp": "2022-11-23T19:28:38+01:00",
          "tree_id": "07b31f94486cdef769f361a41cb3e26424a5ef02",
          "url": "https://github.com/huggingface/safetensors/commit/3836e27dab84717dca64c0c7811eb6a4e4f07231"
        },
        "date": 1669228505930,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0695811787456682,
            "unit": "iter/sec",
            "range": "stddev: 0.02532767476909723",
            "extra": "mean: 934.9453971999878 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9960295866910305,
            "unit": "iter/sec",
            "range": "stddev: 0.06337812446351883",
            "extra": "mean: 333.7750749999941 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.075063666348704,
            "unit": "iter/sec",
            "range": "stddev: 0.012211778029561816",
            "extra": "mean: 325.1965190000078 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 162.93540073007657,
            "unit": "iter/sec",
            "range": "stddev: 0.0019786187414491886",
            "extra": "mean: 6.137401666668059 msec\nrounds: 6"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2115744944908697,
            "unit": "iter/sec",
            "range": "stddev: 0.07810968079674403",
            "extra": "mean: 825.3722776000018 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.296246896909086,
            "unit": "iter/sec",
            "range": "stddev: 0.04374860640529988",
            "extra": "mean: 303.3753329999968 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b5402316fad3d62dd04a9286a31178b7f7f46831",
          "message": "Update workflow. (#109)",
          "timestamp": "2022-11-23T20:27:02+01:00",
          "tree_id": "110076b491129dd32047370e9a6f9e93ab8f2381",
          "url": "https://github.com/huggingface/safetensors/commit/b5402316fad3d62dd04a9286a31178b7f7f46831"
        },
        "date": 1669231979909,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4508155502095952,
            "unit": "iter/sec",
            "range": "stddev: 0.033875894096987755",
            "extra": "mean: 689.26749499999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.326328418090173,
            "unit": "iter/sec",
            "range": "stddev: 0.04850886607712228",
            "extra": "mean: 231.1428775999957 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.628381293682215,
            "unit": "iter/sec",
            "range": "stddev: 0.03992539089799971",
            "extra": "mean: 177.67097639999747 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 229.10893121588123,
            "unit": "iter/sec",
            "range": "stddev: 0.00018694910455792674",
            "extra": "mean: 4.364736000002267 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.959716693885379,
            "unit": "iter/sec",
            "range": "stddev: 0.05817419544916038",
            "extra": "mean: 510.27783920000047 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.872242397466072,
            "unit": "iter/sec",
            "range": "stddev: 0.009295696700881713",
            "extra": "mean: 170.29269779999368 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f4f1cfa88d7c04e6e0f0e19e308cd7b94ab63cfb",
          "message": "I hate yaml. (#110)",
          "timestamp": "2022-11-23T20:57:32+01:00",
          "tree_id": "5c9f74e4038be4370fe634b8b4e168bb2d066340",
          "url": "https://github.com/huggingface/safetensors/commit/f4f1cfa88d7c04e6e0f0e19e308cd7b94ab63cfb"
        },
        "date": 1669233821625,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.222370204885415,
            "unit": "iter/sec",
            "range": "stddev: 0.03560642444111471",
            "extra": "mean: 818.0827674000284 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.113529901788509,
            "unit": "iter/sec",
            "range": "stddev: 0.06561717502731661",
            "extra": "mean: 321.1788650000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.320320062677041,
            "unit": "iter/sec",
            "range": "stddev: 0.010661806115541583",
            "extra": "mean: 301.1757846000364 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 202.00671858308004,
            "unit": "iter/sec",
            "range": "stddev: 0.0008082219852513665",
            "extra": "mean: 4.950330399969971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5952063589495322,
            "unit": "iter/sec",
            "range": "stddev: 0.07149464311199372",
            "extra": "mean: 626.8781429999535 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.152572233212202,
            "unit": "iter/sec",
            "range": "stddev: 0.015672989436674865",
            "extra": "mean: 317.2012965999784 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eef1be5b7fc731baa221e7214cea58221759bed9",
          "message": "New version. (#114)",
          "timestamp": "2022-11-26T22:17:34+01:00",
          "tree_id": "4aa78b841da266fe76bb1474d87db52e7ec41e16",
          "url": "https://github.com/huggingface/safetensors/commit/eef1be5b7fc731baa221e7214cea58221759bed9"
        },
        "date": 1669497890797,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.9983496764699643,
            "unit": "iter/sec",
            "range": "stddev: 0.034395725443958906",
            "extra": "mean: 1.0016530515999875 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6541917728170272,
            "unit": "iter/sec",
            "range": "stddev: 0.07880309843442139",
            "extra": "mean: 376.7625272000032 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.221152169358363,
            "unit": "iter/sec",
            "range": "stddev: 0.03389837309579119",
            "extra": "mean: 310.44792279999456 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 126.88340975304106,
            "unit": "iter/sec",
            "range": "stddev: 0.00025024932578113847",
            "extra": "mean: 7.88125100000343 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3381232070851454,
            "unit": "iter/sec",
            "range": "stddev: 0.10285310211326887",
            "extra": "mean: 747.3153404000186 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.709293250381349,
            "unit": "iter/sec",
            "range": "stddev: 0.05017485808610102",
            "extra": "mean: 369.0999488000216 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patrick.v.platen@gmail.com",
            "name": "Patrick von Platen",
            "username": "patrickvonplaten"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aaac76dfea9dedb43a810b388b5266a889668c13",
          "message": "Add pip installation guide (#116)\n\n* Add pip installation guide\r\n\r\n* Update the 3 locations.\r\n\r\n* Those are not doctests.\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2022-11-28T14:26:04+01:00",
          "tree_id": "f8fd0f705f168692f0cf0825fc121cf309e9b1b9",
          "url": "https://github.com/huggingface/safetensors/commit/aaac76dfea9dedb43a810b388b5266a889668c13"
        },
        "date": 1669642361204,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1947975918704898,
            "unit": "iter/sec",
            "range": "stddev: 0.028515604475666357",
            "extra": "mean: 836.9618476000369 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.028751553237831,
            "unit": "iter/sec",
            "range": "stddev: 0.06784513221774983",
            "extra": "mean: 330.1690423999844 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.9919078279978355,
            "unit": "iter/sec",
            "range": "stddev: 0.0371118463996334",
            "extra": "mean: 250.50678600000538 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 208.6356986200462,
            "unit": "iter/sec",
            "range": "stddev: 0.0005553675541211608",
            "extra": "mean: 4.793043599988778 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5756206828533037,
            "unit": "iter/sec",
            "range": "stddev: 0.05757748026856243",
            "extra": "mean: 634.6705211999961 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.623530425057381,
            "unit": "iter/sec",
            "range": "stddev: 0.031044239286368843",
            "extra": "mean: 275.9739487999923 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "61c8eaf6567429739e6c45fc85c656eeb8f4ef74",
          "message": "Supporting stable-diffusion by default. (#117)",
          "timestamp": "2022-11-30T14:46:38+01:00",
          "tree_id": "0603f4f36731c4739b438c436067a53319f9e376",
          "url": "https://github.com/huggingface/safetensors/commit/61c8eaf6567429739e6c45fc85c656eeb8f4ef74"
        },
        "date": 1669816379604,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2323465813075987,
            "unit": "iter/sec",
            "range": "stddev: 0.04610790868116023",
            "extra": "mean: 811.4600350000046 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0108122035768368,
            "unit": "iter/sec",
            "range": "stddev: 0.07527054275667588",
            "extra": "mean: 332.13629159998845 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.252661052911607,
            "unit": "iter/sec",
            "range": "stddev: 0.0152985903606097",
            "extra": "mean: 307.440579799993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 192.84924264988965,
            "unit": "iter/sec",
            "range": "stddev: 0.0002544462559430576",
            "extra": "mean: 5.185397600007491 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2890616790590606,
            "unit": "iter/sec",
            "range": "stddev: 0.0605211399986295",
            "extra": "mean: 775.758069799997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.9124524121001194,
            "unit": "iter/sec",
            "range": "stddev: 0.013485410646031322",
            "extra": "mean: 343.35324960002254 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd18f427c006e346f4b8e4332e8a2f0e4a6b1fa3",
          "message": "Supporting stable-diffusion by default. (#118)\n\n* New updates.\r\n\r\n* Force load checkpoint on cpu.",
          "timestamp": "2022-11-30T17:01:55+01:00",
          "tree_id": "f7e8691c4b5cbf5df02c001bcd6d25f644237214",
          "url": "https://github.com/huggingface/safetensors/commit/fd18f427c006e346f4b8e4332e8a2f0e4a6b1fa3"
        },
        "date": 1669824511855,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.106230046307808,
            "unit": "iter/sec",
            "range": "stddev: 0.02127665176815707",
            "extra": "mean: 903.9711073999797 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.8620268454432267,
            "unit": "iter/sec",
            "range": "stddev: 0.07887944948984765",
            "extra": "mean: 349.4027323999944 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.2829383798145453,
            "unit": "iter/sec",
            "range": "stddev: 0.013940275564247661",
            "extra": "mean: 304.6051690000013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 200.4436222964819,
            "unit": "iter/sec",
            "range": "stddev: 0.00018174689955208746",
            "extra": "mean: 4.988933988235711 msec\nrounds: 85"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1628709395553518,
            "unit": "iter/sec",
            "range": "stddev: 0.12191793451355398",
            "extra": "mean: 859.9406571999907 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.254761547384709,
            "unit": "iter/sec",
            "range": "stddev: 0.04247891956990929",
            "extra": "mean: 307.24216979997436 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0415b4031e47e232bad0cf018fff355b6866d736",
          "message": "Adding `cargo fuzz` and found sanitation opportunities. (#119)\n\n* Adding `cargo fuzz` and found sanitation opportunities.\r\n\r\n* Better code.",
          "timestamp": "2022-12-05T18:33:12+01:00",
          "tree_id": "4b5895d81f3584307d2d3cd4b45bfddee48c5ef1",
          "url": "https://github.com/huggingface/safetensors/commit/0415b4031e47e232bad0cf018fff355b6866d736"
        },
        "date": 1670261956504,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4792683155012043,
            "unit": "iter/sec",
            "range": "stddev: 0.015461768907321413",
            "extra": "mean: 676.0098823999897 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.707664799751091,
            "unit": "iter/sec",
            "range": "stddev: 0.07875657429239269",
            "extra": "mean: 269.71154460002253 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.042224953159593,
            "unit": "iter/sec",
            "range": "stddev: 0.0381287038337328",
            "extra": "mean: 198.32514599995648 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 212.1978631328368,
            "unit": "iter/sec",
            "range": "stddev: 0.0004548233088580964",
            "extra": "mean: 4.712582800016207 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6644849491227505,
            "unit": "iter/sec",
            "range": "stddev: 0.032389533781754375",
            "extra": "mean: 600.7864477999874 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.5890973495881555,
            "unit": "iter/sec",
            "range": "stddev: 0.019459488569862883",
            "extra": "mean: 178.91976780001642 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "leebutterman@gmail.com",
            "name": "lsb",
            "username": "lsb"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1f01736f7bf5923bbf2b29db84a51e80ff692477",
          "message": "Minor typo (#121)\n\n* Minor typo\r\n\r\n* READMEs.\r\n\r\n* Fix the reamde? ??\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2022-12-06T10:25:11+01:00",
          "tree_id": "5ad70ba2b579fdae1e5641fc0807942e3f7ae3a5",
          "url": "https://github.com/huggingface/safetensors/commit/1f01736f7bf5923bbf2b29db84a51e80ff692477"
        },
        "date": 1670319092444,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3977643371647936,
            "unit": "iter/sec",
            "range": "stddev: 0.008791515008237064",
            "extra": "mean: 715.4281829999945 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.04620639192055,
            "unit": "iter/sec",
            "range": "stddev: 0.06524370745277518",
            "extra": "mean: 247.14507940000203 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.447068404544506,
            "unit": "iter/sec",
            "range": "stddev: 0.011885842007344204",
            "extra": "mean: 224.86724040001036 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 228.88722492878975,
            "unit": "iter/sec",
            "range": "stddev: 0.00016962934354009673",
            "extra": "mean: 4.36896380001599 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8041534901954381,
            "unit": "iter/sec",
            "range": "stddev: 0.11066456369407775",
            "extra": "mean: 554.2765654000277 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.9367294827907555,
            "unit": "iter/sec",
            "range": "stddev: 0.010076581180241993",
            "extra": "mean: 202.5632564000034 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8731ae181db0e6c81e360ccfab89e19de33bd3a6",
          "message": "Python fuzzing. (#123)",
          "timestamp": "2022-12-07T17:04:03+01:00",
          "tree_id": "c34b67588c9799cd55b459301b6b1509eb068d68",
          "url": "https://github.com/huggingface/safetensors/commit/8731ae181db0e6c81e360ccfab89e19de33bd3a6"
        },
        "date": 1670429461612,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0861358490612774,
            "unit": "iter/sec",
            "range": "stddev: 0.015801568393616947",
            "extra": "mean: 920.6951422000088 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9352457719736185,
            "unit": "iter/sec",
            "range": "stddev: 0.08447745454981564",
            "extra": "mean: 340.68697400000474 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.4012282595758965,
            "unit": "iter/sec",
            "range": "stddev: 0.040051460542349905",
            "extra": "mean: 294.01143460000867 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 175.11660927493082,
            "unit": "iter/sec",
            "range": "stddev: 0.0013365748788547013",
            "extra": "mean: 5.710480599987022 msec\nrounds: 20"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3274861212314955,
            "unit": "iter/sec",
            "range": "stddev: 0.08516801395812262",
            "extra": "mean: 753.3035442000028 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.521904270534623,
            "unit": "iter/sec",
            "range": "stddev: 0.021855349814812776",
            "extra": "mean: 283.93730300006155 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f5e1a3d0b82b4d29f69f5c2d547dddba79286f4e",
          "message": "Adding support for Python 3.11 on Linux + fix the source build wheel. (#125)",
          "timestamp": "2022-12-12T10:24:49+01:00",
          "tree_id": "ea105bc1c020048867a2d1a4b35a81cf9f70f968",
          "url": "https://github.com/huggingface/safetensors/commit/f5e1a3d0b82b4d29f69f5c2d547dddba79286f4e"
        },
        "date": 1670837501271,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0823979685754146,
            "unit": "iter/sec",
            "range": "stddev: 0.006059423981839927",
            "extra": "mean: 923.874608999995 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9123707545268216,
            "unit": "iter/sec",
            "range": "stddev: 0.08702002775742583",
            "extra": "mean: 343.3628765999856 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.537620875152735,
            "unit": "iter/sec",
            "range": "stddev: 0.04669045127288663",
            "extra": "mean: 220.37980420000167 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 186.855975189145,
            "unit": "iter/sec",
            "range": "stddev: 0.0003431785512396096",
            "extra": "mean: 5.351715399990553 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3730462962695427,
            "unit": "iter/sec",
            "range": "stddev: 0.011885695555562221",
            "extra": "mean: 728.3075616000133 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.233003626623154,
            "unit": "iter/sec",
            "range": "stddev: 0.055728398174555716",
            "extra": "mean: 309.30989119999595 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1711838d6cfb15d82feb4b60f9a24967e751caff",
          "message": "Keras new format is not safe. (#127)",
          "timestamp": "2022-12-12T11:40:56+01:00",
          "tree_id": "b2a11c30e39a8e72c7906274bcaa1e4c33ad7ade",
          "url": "https://github.com/huggingface/safetensors/commit/1711838d6cfb15d82feb4b60f9a24967e751caff"
        },
        "date": 1670842020181,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4766576971268859,
            "unit": "iter/sec",
            "range": "stddev: 0.034294771270586936",
            "extra": "mean: 677.2050164000007 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.0385959438664845,
            "unit": "iter/sec",
            "range": "stddev: 0.06637953885714479",
            "extra": "mean: 247.61080680000305 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 7.553920907214739,
            "unit": "iter/sec",
            "range": "stddev: 0.022045213482863212",
            "extra": "mean: 132.38158200000498 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 228.51009907289608,
            "unit": "iter/sec",
            "range": "stddev: 0.00018685088735647437",
            "extra": "mean: 4.376174199990146 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0532186445805403,
            "unit": "iter/sec",
            "range": "stddev: 0.057209471115363605",
            "extra": "mean: 487.04019060000974 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.458478373564162,
            "unit": "iter/sec",
            "range": "stddev: 0.0158297810618179",
            "extra": "mean: 183.20123880000665 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "706d70a42f71a8910271e90ea0f0557691c943b8",
          "message": "Release rust. (#126)",
          "timestamp": "2022-12-12T14:59:19+01:00",
          "tree_id": "2206f93df126b1716b5525bf95f6a8a1d81eb2f1",
          "url": "https://github.com/huggingface/safetensors/commit/706d70a42f71a8910271e90ea0f0557691c943b8"
        },
        "date": 1670853940426,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.073815060511454,
            "unit": "iter/sec",
            "range": "stddev: 0.027067624785834257",
            "extra": "mean: 931.2590563999947 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.769205781281101,
            "unit": "iter/sec",
            "range": "stddev: 0.09923244184342665",
            "extra": "mean: 361.1143695999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.913693450516344,
            "unit": "iter/sec",
            "range": "stddev: 0.057761030683026154",
            "extra": "mean: 255.51311379997514 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 211.9118085758628,
            "unit": "iter/sec",
            "range": "stddev: 0.00018732464618013875",
            "extra": "mean: 4.7189442000444615 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.377107693173312,
            "unit": "iter/sec",
            "range": "stddev: 0.07016261433147898",
            "extra": "mean: 726.1596205999467 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.9271750810773924,
            "unit": "iter/sec",
            "range": "stddev: 0.015484126042556971",
            "extra": "mean: 254.63596080001028 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "02e5f4abf7d11a916cad15f5add0a7df61b1fdb8",
          "message": "Necessary modifications to publish on Rust. (#129)\n\n* Adding Info to publish.\r\n\r\n* Bench.\r\n\r\n* Update README.",
          "timestamp": "2022-12-13T16:14:56+01:00",
          "tree_id": "9e4050140657a1bbbf2b45148d58c4a3fbdba40a",
          "url": "https://github.com/huggingface/safetensors/commit/02e5f4abf7d11a916cad15f5add0a7df61b1fdb8"
        },
        "date": 1670944934022,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4734209134351193,
            "unit": "iter/sec",
            "range": "stddev: 0.019208149957301995",
            "extra": "mean: 678.6926877999917 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.7262471123078345,
            "unit": "iter/sec",
            "range": "stddev: 0.07697304739664311",
            "extra": "mean: 268.36652799997864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.700160936612473,
            "unit": "iter/sec",
            "range": "stddev: 0.048513278350684724",
            "extra": "mean: 212.75867220000464 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 225.83608130644603,
            "unit": "iter/sec",
            "range": "stddev: 0.00018480419601286805",
            "extra": "mean: 4.427990400006365 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0943582138641883,
            "unit": "iter/sec",
            "range": "stddev: 0.047839866144771016",
            "extra": "mean: 477.4732390000054 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.506608293388006,
            "unit": "iter/sec",
            "range": "stddev: 0.015664637358197207",
            "extra": "mean: 181.59998800000685 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "846a64b937df6818787e7e0bdf88eaa97330f09d",
          "message": "Fixing it for `diffusers`. (#130)",
          "timestamp": "2022-12-13T17:17:42+01:00",
          "tree_id": "24c5256391a1a21b31e08e7b7743649f03bb75d2",
          "url": "https://github.com/huggingface/safetensors/commit/846a64b937df6818787e7e0bdf88eaa97330f09d"
        },
        "date": 1670948624288,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1286691287607022,
            "unit": "iter/sec",
            "range": "stddev: 0.022539950978805772",
            "extra": "mean: 885.999248600001 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.72617063908647,
            "unit": "iter/sec",
            "range": "stddev: 0.10842052053253164",
            "extra": "mean: 366.8148962000032 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.460294728070529,
            "unit": "iter/sec",
            "range": "stddev: 0.05679341566730655",
            "extra": "mean: 288.9927241999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 214.6473736456541,
            "unit": "iter/sec",
            "range": "stddev: 0.00040661447829041234",
            "extra": "mean: 4.658803799998168 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.362354685118712,
            "unit": "iter/sec",
            "range": "stddev: 0.042417146903536845",
            "extra": "mean: 734.0232399999877 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7137618168773967,
            "unit": "iter/sec",
            "range": "stddev: 0.031730474527720036",
            "extra": "mean: 269.268749399987 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "24695242+thomasw21@users.noreply.github.com",
            "name": "Thomas Wang",
            "username": "thomasw21"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fda385b693ba5710a25b715316a8c3780e4e81b9",
          "message": "Update README.md (#131)\n\n* Update README.md\r\n\r\n* Update other files.\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2022-12-14T11:23:50+01:00",
          "tree_id": "3fce743dd4a41671730f4f2fd22c40351a41a765",
          "url": "https://github.com/huggingface/safetensors/commit/fda385b693ba5710a25b715316a8c3780e4e81b9"
        },
        "date": 1671013824883,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4181497282278384,
            "unit": "iter/sec",
            "range": "stddev: 0.015614682087880206",
            "extra": "mean: 705.1441608000232 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.8874785112525836,
            "unit": "iter/sec",
            "range": "stddev: 0.07266546049174463",
            "extra": "mean: 257.2361486000318 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 6.790889123669347,
            "unit": "iter/sec",
            "range": "stddev: 0.02074978436135855",
            "extra": "mean: 147.25612240001738 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 229.3646670814444,
            "unit": "iter/sec",
            "range": "stddev: 0.00015897415169922148",
            "extra": "mean: 4.35986942855899 msec\nrounds: 7"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0537071436639494,
            "unit": "iter/sec",
            "range": "stddev: 0.0463277657229029",
            "extra": "mean: 486.9243421999954 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 6.551969167376258,
            "unit": "iter/sec",
            "range": "stddev: 0.00790534044363028",
            "extra": "mean: 152.6258708571504 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "41013271a9b5e9d0af615606e0803dfc9c75ae96",
          "message": "New version. (#132)",
          "timestamp": "2022-12-14T13:55:28+01:00",
          "tree_id": "02ec7f3b963d4b25453397aa94b1bb2aeea62ebe",
          "url": "https://github.com/huggingface/safetensors/commit/41013271a9b5e9d0af615606e0803dfc9c75ae96"
        },
        "date": 1671022898543,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1679655068353423,
            "unit": "iter/sec",
            "range": "stddev: 0.01715909189775859",
            "extra": "mean: 856.1896684000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3129669726750275,
            "unit": "iter/sec",
            "range": "stddev: 0.06716207186668016",
            "extra": "mean: 301.84424060000765 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.028241865144873,
            "unit": "iter/sec",
            "range": "stddev: 0.04387817685670329",
            "extra": "mean: 248.24725860000854 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 216.85504068874283,
            "unit": "iter/sec",
            "range": "stddev: 0.0002472169734390966",
            "extra": "mean: 4.61137540000891 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4288556683800901,
            "unit": "iter/sec",
            "range": "stddev: 0.10978383445656126",
            "extra": "mean: 699.8607501999913 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.025665893768912,
            "unit": "iter/sec",
            "range": "stddev: 0.012205905496974478",
            "extra": "mean: 248.40610879999758 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vilem.zouhar@gmail.com",
            "name": "Vilém Zouhar",
            "username": "zouharvi"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "23d5934f06704e028ac3310edb44bfecef204d8a",
          "message": "Fix readme typos (#133)\n\n* fix readme typos\r\n\r\n* Replicating docs on all 3 files.\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2022-12-15T09:38:37+01:00",
          "tree_id": "03d1980920f207e9b0ee8efca868da1955557b77",
          "url": "https://github.com/huggingface/safetensors/commit/23d5934f06704e028ac3310edb44bfecef204d8a"
        },
        "date": 1671093872286,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4130069586966871,
            "unit": "iter/sec",
            "range": "stddev: 0.038469352360525046",
            "extra": "mean: 707.7105982000035 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.111715930362102,
            "unit": "iter/sec",
            "range": "stddev: 0.06407732573446007",
            "extra": "mean: 243.20746299998746 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 6.700684253774241,
            "unit": "iter/sec",
            "range": "stddev: 0.020332982433804995",
            "extra": "mean: 149.23848999999336 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 236.07503421096047,
            "unit": "iter/sec",
            "range": "stddev: 0.00009043357971264568",
            "extra": "mean: 4.235941353740879 msec\nrounds: 147"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7130571285078198,
            "unit": "iter/sec",
            "range": "stddev: 0.05869361258671062",
            "extra": "mean: 583.7516936000043 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.1303663875124785,
            "unit": "iter/sec",
            "range": "stddev: 0.010457080450674084",
            "extra": "mean: 242.10927220000258 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ec48e3f894f1cc73827bc8ee9ff580d677e40dd0",
          "message": "Adding stale bot. (#135)\n\n* Adding stale bot.\r\n\r\n* Clippy.",
          "timestamp": "2022-12-19T13:50:15+01:00",
          "tree_id": "692f0e1041678f4fa52e25136ee2fbf113add5db",
          "url": "https://github.com/huggingface/safetensors/commit/ec48e3f894f1cc73827bc8ee9ff580d677e40dd0"
        },
        "date": 1671454582073,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3734602717180702,
            "unit": "iter/sec",
            "range": "stddev: 0.04917498420041616",
            "extra": "mean: 728.0880420000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.345448354890999,
            "unit": "iter/sec",
            "range": "stddev: 0.06675781518963446",
            "extra": "mean: 298.9135965999935 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.812250737891638,
            "unit": "iter/sec",
            "range": "stddev: 0.026017069953805788",
            "extra": "mean: 262.3122319999993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 216.74196426954362,
            "unit": "iter/sec",
            "range": "stddev: 0.00020435361020965638",
            "extra": "mean: 4.613781200009726 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6153185470170586,
            "unit": "iter/sec",
            "range": "stddev: 0.044803422146665356",
            "extra": "mean: 619.0729388000022 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.869124924163465,
            "unit": "iter/sec",
            "range": "stddev: 0.04758305347474547",
            "extra": "mean: 258.4563744000093 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a104c5e153364d774569f47e4d7666ba1664ab03",
          "message": "Making SAFETENSORS_FAST_GPU=1 work on Windows again. (#140)\n\n* Making Windows load cudart differently.\r\n\r\n* Remove unsafe.\r\n\r\n* Misplaced comment",
          "timestamp": "2022-12-27T09:13:28+01:00",
          "tree_id": "3bbb018a0af1dd8ad4a9c3f632a63813a0e80480",
          "url": "https://github.com/huggingface/safetensors/commit/a104c5e153364d774569f47e4d7666ba1664ab03"
        },
        "date": 1672129111401,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2371303895711236,
            "unit": "iter/sec",
            "range": "stddev: 0.014661077028059464",
            "extra": "mean: 808.3222338000041 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.232427691848192,
            "unit": "iter/sec",
            "range": "stddev: 0.07822849142440552",
            "extra": "mean: 309.36500220001335 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.5228681415753,
            "unit": "iter/sec",
            "range": "stddev: 0.009796546510707872",
            "extra": "mean: 283.85961660002295 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 217.9344538593799,
            "unit": "iter/sec",
            "range": "stddev: 0.0002053464775600215",
            "extra": "mean: 4.588535599998522 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.554805685488301,
            "unit": "iter/sec",
            "range": "stddev: 0.09679703485883735",
            "extra": "mean: 643.1671875999996 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.530220917920765,
            "unit": "iter/sec",
            "range": "stddev: 0.012013521233173792",
            "extra": "mean: 283.2683911999993 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "alvarobartt@yahoo.com",
            "name": "Alvaro Bartolome",
            "username": "alvarobartt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ba8de5ff52eee70e6bbc7e7c49a86c251da357d7",
          "message": "Update docstrings & type-hints (#139)",
          "timestamp": "2022-12-27T10:02:52+01:00",
          "tree_id": "e504d4f57f2d4de1015f2c4ef0cbc3cf7909f273",
          "url": "https://github.com/huggingface/safetensors/commit/ba8de5ff52eee70e6bbc7e7c49a86c251da357d7"
        },
        "date": 1672132111227,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0714858621704049,
            "unit": "iter/sec",
            "range": "stddev: 0.010086055127042914",
            "extra": "mean: 933.2834293999895 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.738221282528031,
            "unit": "iter/sec",
            "range": "stddev: 0.07770379941033304",
            "extra": "mean: 365.2005798000232 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.275355981338185,
            "unit": "iter/sec",
            "range": "stddev: 0.02963345779290683",
            "extra": "mean: 233.89865179998424 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.09748839225335,
            "unit": "iter/sec",
            "range": "stddev: 0.0003886173976730558",
            "extra": "mean: 7.627911200006565 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.361570791676603,
            "unit": "iter/sec",
            "range": "stddev: 0.031568333453674546",
            "extra": "mean: 734.4458372000076 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6405967024497894,
            "unit": "iter/sec",
            "range": "stddev: 0.01841661828047926",
            "extra": "mean: 274.6802465999849 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6cd64c7ef74bbacaa2fedb5d403e83bc98f281a0",
          "message": "Fixing torch version parsing. (#143)\n\n* Fixing torch version parsing.\r\n\r\n* Anticipating a bit more variations of this.",
          "timestamp": "2022-12-29T08:17:56+01:00",
          "tree_id": "bead54ea87c92a4a991cf23b9ed8310f452e6fe9",
          "url": "https://github.com/huggingface/safetensors/commit/6cd64c7ef74bbacaa2fedb5d403e83bc98f281a0"
        },
        "date": 1672298605342,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1744215042951465,
            "unit": "iter/sec",
            "range": "stddev: 0.017178312549373853",
            "extra": "mean: 851.483046199985 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0557286016358,
            "unit": "iter/sec",
            "range": "stddev: 0.0756391051884622",
            "extra": "mean: 327.2541938000245 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.117969165887379,
            "unit": "iter/sec",
            "range": "stddev: 0.03603051448908672",
            "extra": "mean: 242.83814659999052 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 186.38844686873932,
            "unit": "iter/sec",
            "range": "stddev: 0.00019274483364365092",
            "extra": "mean: 5.3651393999984975 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3972855265511843,
            "unit": "iter/sec",
            "range": "stddev: 0.05210721152604907",
            "extra": "mean: 715.6733401999986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.199270998241848,
            "unit": "iter/sec",
            "range": "stddev: 0.011342745205001777",
            "extra": "mean: 312.5712078000106 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dc891f56afc61fbbbc2e804ad8ed156a0798a75c",
          "message": "Updating rust ergonomics. (#144)\n\n- `get_` prefix removed in a few functions.\r\n- Make error implement Error.",
          "timestamp": "2022-12-30T14:28:44+01:00",
          "tree_id": "aa1baae8a3825006b33fadc5f4a941ad5a7224e1",
          "url": "https://github.com/huggingface/safetensors/commit/dc891f56afc61fbbbc2e804ad8ed156a0798a75c"
        },
        "date": 1672407255015,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1341784657516945,
            "unit": "iter/sec",
            "range": "stddev: 0.027611115869293164",
            "extra": "mean: 881.6954564000071 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.7456486729195295,
            "unit": "iter/sec",
            "range": "stddev: 0.0836491988921162",
            "extra": "mean: 364.2126576000237 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.177408273279422,
            "unit": "iter/sec",
            "range": "stddev: 0.0305407553241278",
            "extra": "mean: 239.3828744000075 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 175.20772803533112,
            "unit": "iter/sec",
            "range": "stddev: 0.00018395171567083294",
            "extra": "mean: 5.707510799970805 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3670172231598852,
            "unit": "iter/sec",
            "range": "stddev: 0.049910284929194505",
            "extra": "mean: 731.5196788000094 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6309968223649194,
            "unit": "iter/sec",
            "range": "stddev: 0.015751702175088863",
            "extra": "mean: 275.40646520001246 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e8b3ab601fcd9859567cc9e1c11301a5647d3ab5",
          "message": "Preparing next version. (#158)",
          "timestamp": "2023-01-16T10:34:28+01:00",
          "tree_id": "ab87937f1900366a3b3dcc948710704ba1845ef4",
          "url": "https://github.com/huggingface/safetensors/commit/e8b3ab601fcd9859567cc9e1c11301a5647d3ab5"
        },
        "date": 1673861969107,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.257003466462589,
            "unit": "iter/sec",
            "range": "stddev: 0.049299663691434584",
            "extra": "mean: 795.5427544000031 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3583468920542368,
            "unit": "iter/sec",
            "range": "stddev: 0.0617306732763601",
            "extra": "mean: 297.76554720001513 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.853994566530183,
            "unit": "iter/sec",
            "range": "stddev: 0.025612292592708766",
            "extra": "mean: 206.0158877999811 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 219.1138440144726,
            "unit": "iter/sec",
            "range": "stddev: 0.00023436566157043613",
            "extra": "mean: 4.563837600028364 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4982723471008825,
            "unit": "iter/sec",
            "range": "stddev: 0.06366293732213446",
            "extra": "mean: 667.4353977999886 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.817969002784647,
            "unit": "iter/sec",
            "range": "stddev: 0.011275060470258925",
            "extra": "mean: 261.9193606000067 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "02e57076007ecf318ff3a01865e1be07874cb979",
          "message": "Fix byte order detection for `np.uint8` and other types for which byte order is not defined. (#160)\n\n* Fix byte order detection for `np.uint8` and other types for which byte order is not defined.\r\n\r\n* Fixing np.uint8 endianness for `save`.\r\n\r\nCo-authored-by: KOLANICH <kolan_n@mail.ru>",
          "timestamp": "2023-01-16T13:42:22+01:00",
          "tree_id": "b17fdd027bce9f0435ecc0ee58c809d204c7a089",
          "url": "https://github.com/huggingface/safetensors/commit/02e57076007ecf318ff3a01865e1be07874cb979"
        },
        "date": 1673873228073,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3005023887805756,
            "unit": "iter/sec",
            "range": "stddev: 0.034029487385492205",
            "extra": "mean: 768.9336126000171 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3847954451097904,
            "unit": "iter/sec",
            "range": "stddev: 0.06631660413546979",
            "extra": "mean: 295.43882820001954 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.363074771498253,
            "unit": "iter/sec",
            "range": "stddev: 0.02985854970954602",
            "extra": "mean: 229.1961638000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 213.29179887868105,
            "unit": "iter/sec",
            "range": "stddev: 0.0005053952973302503",
            "extra": "mean: 4.6884128000101555 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6664924882047836,
            "unit": "iter/sec",
            "range": "stddev: 0.04789959972952724",
            "extra": "mean: 600.0627108000003 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.055408984828815,
            "unit": "iter/sec",
            "range": "stddev: 0.016126474924252725",
            "extra": "mean: 246.58425420000185 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "24695242+thomasw21@users.noreply.github.com",
            "name": "Thomas Wang",
            "username": "thomasw21"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "671c40eee71afea2f3b969ead5bb1dfc496ab06f",
          "message": "Improve type hinting (#159)",
          "timestamp": "2023-01-16T15:28:18+01:00",
          "tree_id": "e4b8fa0d430d383434374b9d8d1e51cccca1f248",
          "url": "https://github.com/huggingface/safetensors/commit/671c40eee71afea2f3b969ead5bb1dfc496ab06f"
        },
        "date": 1673879638448,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1151122868907608,
            "unit": "iter/sec",
            "range": "stddev: 0.02420453454650197",
            "extra": "mean: 896.7706765999992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.68813202741958,
            "unit": "iter/sec",
            "range": "stddev: 0.07365752512983258",
            "extra": "mean: 372.0055376000005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.0238366399493173,
            "unit": "iter/sec",
            "range": "stddev: 0.03989801019488469",
            "extra": "mean: 330.7056958000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.48539732742785,
            "unit": "iter/sec",
            "range": "stddev: 0.0005325167402561757",
            "extra": "mean: 7.663692799974342 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2134114143872807,
            "unit": "iter/sec",
            "range": "stddev: 0.15258249932560286",
            "extra": "mean: 824.122789799992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.4394226671508226,
            "unit": "iter/sec",
            "range": "stddev: 0.015114249784933783",
            "extra": "mean: 290.7464702000084 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e23d45b5d2cd0e37bb5c40607cf9f373baef9ab6",
          "message": "New unreleased version. (#161)\n\n* New unreleased version.\r\n\r\n* Making only 1 release now.",
          "timestamp": "2023-01-16T16:40:29+01:00",
          "tree_id": "1c6c888a84ef04e7b33248033cb5df5231ff9901",
          "url": "https://github.com/huggingface/safetensors/commit/e23d45b5d2cd0e37bb5c40607cf9f373baef9ab6"
        },
        "date": 1673883918487,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4700131899139006,
            "unit": "iter/sec",
            "range": "stddev: 0.018442066592634535",
            "extra": "mean: 680.2660049999758 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.9421768594176294,
            "unit": "iter/sec",
            "range": "stddev: 0.06979867198220203",
            "extra": "mean: 253.66695500001697 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.5395733202597555,
            "unit": "iter/sec",
            "range": "stddev: 0.029851502673670326",
            "extra": "mean: 180.51931839997906 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 221.97308386123723,
            "unit": "iter/sec",
            "range": "stddev: 0.00015768398746574276",
            "extra": "mean: 4.505050714279995 msec\nrounds: 7"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.004384239355521,
            "unit": "iter/sec",
            "range": "stddev: 0.06337372986126737",
            "extra": "mean: 498.90633759998764 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.2556950428989815,
            "unit": "iter/sec",
            "range": "stddev: 0.02032911439279166",
            "extra": "mean: 190.2697915000052 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "24695242+thomasw21@users.noreply.github.com",
            "name": "Thomas Wang",
            "username": "thomasw21"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "af4b32fb09810d0c407e7b3d277730f1409a8a1d",
          "message": "Update README.md (#163)\n\n* Update README.md\r\n\r\nLink to huggingface.co documentation\r\n\r\n* Make rust doc match\r\n\r\n* Why is the documentation at three different places?",
          "timestamp": "2023-01-17T10:02:28+01:00",
          "tree_id": "50c5465352ab3a28d91c87dab71b95abe99fdbec",
          "url": "https://github.com/huggingface/safetensors/commit/af4b32fb09810d0c407e7b3d277730f1409a8a1d"
        },
        "date": 1673946481488,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1181854015024812,
            "unit": "iter/sec",
            "range": "stddev: 0.014381519296053967",
            "extra": "mean: 894.3060772000081 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.076768797321842,
            "unit": "iter/sec",
            "range": "stddev: 0.07734033519590326",
            "extra": "mean: 325.01629660000617 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.757422985577982,
            "unit": "iter/sec",
            "range": "stddev: 0.023423467748034544",
            "extra": "mean: 173.68881920000376 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 190.9353007713981,
            "unit": "iter/sec",
            "range": "stddev: 0.0003143112245726478",
            "extra": "mean: 5.237376200000199 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3338433569740271,
            "unit": "iter/sec",
            "range": "stddev: 0.02516575608568927",
            "extra": "mean: 749.7132214000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.8666265406604854,
            "unit": "iter/sec",
            "range": "stddev: 0.008915174303975995",
            "extra": "mean: 348.84209219998183 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "225ed02b51c85050c35f5f4fde4fa1f769e8b457",
          "message": "Enabling 3.8 support for M1. (#162)",
          "timestamp": "2023-01-17T10:04:04+01:00",
          "tree_id": "347bf57358844db526038fdb0d473af51f8c94ce",
          "url": "https://github.com/huggingface/safetensors/commit/225ed02b51c85050c35f5f4fde4fa1f769e8b457"
        },
        "date": 1673946560019,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1270458150568319,
            "unit": "iter/sec",
            "range": "stddev: 0.01889542130535171",
            "extra": "mean: 887.275376599996 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.7868220557623244,
            "unit": "iter/sec",
            "range": "stddev: 0.08742289701925253",
            "extra": "mean: 358.8316655999961 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.154485032719015,
            "unit": "iter/sec",
            "range": "stddev: 0.0401852088131597",
            "extra": "mean: 317.00895379999565 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 235.9921992308295,
            "unit": "iter/sec",
            "range": "stddev: 0.00024181735341456573",
            "extra": "mean: 4.237428199996884 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2607963447218256,
            "unit": "iter/sec",
            "range": "stddev: 0.10160531351077172",
            "extra": "mean: 793.1495076000033 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.565988253899691,
            "unit": "iter/sec",
            "range": "stddev: 0.018921241654401823",
            "extra": "mean: 280.4271716000244 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "KOLANICH@users.noreply.github.com",
            "name": "KOLANICH",
            "username": "KOLANICH"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2a5cac88d9ee4af7f4430a0db09000945924b4e7",
          "message": "Added JSONSchema of the header. (#151)",
          "timestamp": "2023-01-17T17:39:47+01:00",
          "tree_id": "07c14496585d17298ab43a0a0a246bfe7a34d2ae",
          "url": "https://github.com/huggingface/safetensors/commit/2a5cac88d9ee4af7f4430a0db09000945924b4e7"
        },
        "date": 1673973886200,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2681830378257404,
            "unit": "iter/sec",
            "range": "stddev: 0.025868413422122614",
            "extra": "mean: 788.5297075999915 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3142001180038214,
            "unit": "iter/sec",
            "range": "stddev: 0.06974056805787486",
            "extra": "mean: 301.73193059998766 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.253003899024226,
            "unit": "iter/sec",
            "range": "stddev: 0.024753901204505355",
            "extra": "mean: 190.36726780000208 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 215.396433690001,
            "unit": "iter/sec",
            "range": "stddev: 0.00019193853986325187",
            "extra": "mean: 4.642602399997031 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3358675042685886,
            "unit": "iter/sec",
            "range": "stddev: 0.0960079584853439",
            "extra": "mean: 748.5772330000032 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.944416600281851,
            "unit": "iter/sec",
            "range": "stddev: 0.026452204835505266",
            "extra": "mean: 253.5229163999929 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "613a34f4ec0a55f4dd64734f640b6aea852aaa45",
          "message": "Adding custom error type for users to properly catch. (#165)",
          "timestamp": "2023-01-17T18:06:34+01:00",
          "tree_id": "989f4d6e6b65525944fa4e93e299ef106970c7c7",
          "url": "https://github.com/huggingface/safetensors/commit/613a34f4ec0a55f4dd64734f640b6aea852aaa45"
        },
        "date": 1673975488824,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4032726506758237,
            "unit": "iter/sec",
            "range": "stddev: 0.020943359134530798",
            "extra": "mean: 712.619888599977 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.09574115398236,
            "unit": "iter/sec",
            "range": "stddev: 0.06416546284747063",
            "extra": "mean: 244.15605439999126 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.127879031914812,
            "unit": "iter/sec",
            "range": "stddev: 0.023939970824437894",
            "extra": "mean: 195.01240059997826 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 221.04565026005727,
            "unit": "iter/sec",
            "range": "stddev: 0.00017720310955706594",
            "extra": "mean: 4.523952399983955 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8159043307799785,
            "unit": "iter/sec",
            "range": "stddev: 0.05525210895165633",
            "extra": "mean: 550.6898040000124 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.677764058267803,
            "unit": "iter/sec",
            "range": "stddev: 0.007147328569720978",
            "extra": "mean: 176.1256702000196 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a4042652348df70ddac5ade96698296365a42073",
          "message": "Fixing the fact that the context manager wasn't properly cleaning up after itself. (#166)\n\n* Adding failing test for windows.\r\n\r\n* This should fail.\r\n\r\n* Fixing Windows issue by actually dropping the rust resources.\r\n\r\n* Making the rust module private.",
          "timestamp": "2023-01-19T14:30:56+01:00",
          "tree_id": "169e0f5af0623fd9897c78ece59ed44fc4428f71",
          "url": "https://github.com/huggingface/safetensors/commit/a4042652348df70ddac5ade96698296365a42073"
        },
        "date": 1674135359012,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.427040932471809,
            "unit": "iter/sec",
            "range": "stddev: 0.01525198702415532",
            "extra": "mean: 700.7507473999908 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.4709517478256693,
            "unit": "iter/sec",
            "range": "stddev: 0.0927009310552714",
            "extra": "mean: 288.10541679999915 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.99142557441404,
            "unit": "iter/sec",
            "range": "stddev: 0.02334475898722056",
            "extra": "mean: 200.34356620000153 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 216.49074051706475,
            "unit": "iter/sec",
            "range": "stddev: 0.00019798116268080103",
            "extra": "mean: 4.619135200016444 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.2281325339023152,
            "unit": "iter/sec",
            "range": "stddev: 0.0633636376122456",
            "extra": "mean: 448.80633660001195 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.579982911716987,
            "unit": "iter/sec",
            "range": "stddev: 0.020595025337002682",
            "extra": "mean: 218.3414260000177 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jnelso11@gmu.edu",
            "name": "John B Nelson",
            "username": "jbn"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bcb033f2e0c18c2737b7832140ccebfe5958d0a1",
          "message": "Minor pydoc fixes for torch bindings (#167)",
          "timestamp": "2023-01-20T11:07:11+01:00",
          "tree_id": "d93a5814a5013133cf1d1d85778999e4076fe18b",
          "url": "https://github.com/huggingface/safetensors/commit/bcb033f2e0c18c2737b7832140ccebfe5958d0a1"
        },
        "date": 1674209516500,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.5081962951792334,
            "unit": "iter/sec",
            "range": "stddev: 0.027558008347608562",
            "extra": "mean: 663.0436656000143 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.43902281138139,
            "unit": "iter/sec",
            "range": "stddev: 0.05237290990175643",
            "extra": "mean: 225.27480539997669 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.322154628658416,
            "unit": "iter/sec",
            "range": "stddev: 0.021546221533814025",
            "extra": "mean: 187.893826800007 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 218.11304991267784,
            "unit": "iter/sec",
            "range": "stddev: 0.00017307837928983973",
            "extra": "mean: 4.5847784000102365 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.879089408726132,
            "unit": "iter/sec",
            "range": "stddev: 0.09986745742909539",
            "extra": "mean: 532.1726552000086 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.979150058000217,
            "unit": "iter/sec",
            "range": "stddev: 0.011773647756242747",
            "extra": "mean: 167.24785133331466 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a98326ad6ec0ee702a92c8bad15033aecbebaa02",
          "message": "Better error messages on outdated context manager. (#168)\n\n* Better error messages on outdated context manager.\r\n\r\n* Clippy.\r\n\r\n* Remove unwrap.\r\n\r\n* Fix.",
          "timestamp": "2023-01-20T15:05:24+01:00",
          "tree_id": "d3a98444262fcb5e26b30140a16ba621410c66ae",
          "url": "https://github.com/huggingface/safetensors/commit/a98326ad6ec0ee702a92c8bad15033aecbebaa02"
        },
        "date": 1674223814479,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3903651210287353,
            "unit": "iter/sec",
            "range": "stddev: 0.016515687979899604",
            "extra": "mean: 719.235533799997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.9290521528735693,
            "unit": "iter/sec",
            "range": "stddev: 0.06727934653203052",
            "extra": "mean: 254.51431059998413 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.746788789820479,
            "unit": "iter/sec",
            "range": "stddev: 0.02189928209110637",
            "extra": "mean: 174.01022320001402 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 227.86190157108797,
            "unit": "iter/sec",
            "range": "stddev: 0.00007546302697308547",
            "extra": "mean: 4.388623078737986 msec\nrounds: 127"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.001474898861815,
            "unit": "iter/sec",
            "range": "stddev: 0.08699507182480477",
            "extra": "mean: 499.6315470000013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.066735401218483,
            "unit": "iter/sec",
            "range": "stddev: 0.010818505771205932",
            "extra": "mean: 245.89748320000808 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9a761ac0e01413d0d88bc4f12c950669ec2ead0",
          "message": "Clippy fix. (#171)\n\n* Clippy fix.\r\n\r\n* And tests.",
          "timestamp": "2023-01-30T13:59:40+01:00",
          "tree_id": "d23f493fc5e8ebb9d2b3faf35751c7ff9ca13478",
          "url": "https://github.com/huggingface/safetensors/commit/f9a761ac0e01413d0d88bc4f12c950669ec2ead0"
        },
        "date": 1675083883778,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4308576340936112,
            "unit": "iter/sec",
            "range": "stddev: 0.009107754105040831",
            "extra": "mean: 698.8815492000072 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.631252021926557,
            "unit": "iter/sec",
            "range": "stddev: 0.08798092198824208",
            "extra": "mean: 275.38710999999694 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.9020364682282853,
            "unit": "iter/sec",
            "range": "stddev: 0.01299414534617922",
            "extra": "mean: 256.27643620000526 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 229.02546960208434,
            "unit": "iter/sec",
            "range": "stddev: 0.00019521601003369277",
            "extra": "mean: 4.366326599995318 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7053930079329263,
            "unit": "iter/sec",
            "range": "stddev: 0.03566858910572391",
            "extra": "mean: 586.3751026000045 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.069334422426101,
            "unit": "iter/sec",
            "range": "stddev: 0.04841518427397529",
            "extra": "mean: 197.2645552000131 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3cf73afac4d5156aac4adfa36acd407f0882a3f9",
          "message": "Update README.md (#170)\n\nFixes #169 169",
          "timestamp": "2023-01-30T14:00:51+01:00",
          "tree_id": "5e38940e85f4418ae0d799d1bdc4da662401c93e",
          "url": "https://github.com/huggingface/safetensors/commit/3cf73afac4d5156aac4adfa36acd407f0882a3f9"
        },
        "date": 1675083978956,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2469398422255207,
            "unit": "iter/sec",
            "range": "stddev: 0.03295445964008003",
            "extra": "mean: 801.9633073999898 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3407554317694808,
            "unit": "iter/sec",
            "range": "stddev: 0.07145830878292474",
            "extra": "mean: 299.33349520001684 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.5674662809133784,
            "unit": "iter/sec",
            "range": "stddev: 0.009836144098266521",
            "extra": "mean: 280.31098860000156 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 216.54577781992631,
            "unit": "iter/sec",
            "range": "stddev: 0.00021646381576670984",
            "extra": "mean: 4.617961200017362 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3328717086783388,
            "unit": "iter/sec",
            "range": "stddev: 0.04217502140361372",
            "extra": "mean: 750.2597538000032 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.138831671656736,
            "unit": "iter/sec",
            "range": "stddev: 0.033456692742512145",
            "extra": "mean: 241.6140783999822 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3dc566856400ad2f3b854573327816a3360fa7ac",
          "message": "Fixing error message for FileNotFound. (#175)\n\n* Fixing error message for FileNotFound.\r\n\r\n* Old PR readme fix.",
          "timestamp": "2023-02-06T12:05:40+01:00",
          "tree_id": "a4ab64394ca5b949407fbaedb614642216c9c709",
          "url": "https://github.com/huggingface/safetensors/commit/3dc566856400ad2f3b854573327816a3360fa7ac"
        },
        "date": 1675681836964,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4720054336359853,
            "unit": "iter/sec",
            "range": "stddev: 0.01268973078579214",
            "extra": "mean: 679.3453184000214 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.8509222358013204,
            "unit": "iter/sec",
            "range": "stddev: 0.07717548122885173",
            "extra": "mean: 259.6780559999843 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.49758600976158,
            "unit": "iter/sec",
            "range": "stddev: 0.04493664747836575",
            "extra": "mean: 222.34149559999423 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 215.7515922468206,
            "unit": "iter/sec",
            "range": "stddev: 0.0003950355214495014",
            "extra": "mean: 4.6349599999985 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.813714418625376,
            "unit": "iter/sec",
            "range": "stddev: 0.06168254183591069",
            "extra": "mean: 551.3547169999924 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.364955992006377,
            "unit": "iter/sec",
            "range": "stddev: 0.02584184039063614",
            "extra": "mean: 186.39481879999948 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "50394665+JunnYu@users.noreply.github.com",
            "name": "yujun",
            "username": "JunnYu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c4e244ece49351ef67c8c132bf50b621133499d1",
          "message": "Add PaddlePaddle backend (#176)\n\n* add paddle backend\r\n\r\n* add paddle ace\r\n\r\n* update docs\r\n\r\n* Some modifications.\r\n\r\n* Consistency.\r\n\r\n* Showing Paddle format.\r\n\r\n* donot modify lib.rs\r\n\r\n* remove test_deserialization_safe_open test\r\n\r\n* rm unused import\r\n\r\n* make style\r\n\r\n* Style.\r\n\r\n---------\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2023-02-07T15:44:54+01:00",
          "tree_id": "fc0ac6a930691de94aed0d7276a58fcbdd48c4bb",
          "url": "https://github.com/huggingface/safetensors/commit/c4e244ece49351ef67c8c132bf50b621133499d1"
        },
        "date": 1675781427158,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2568768559194343,
            "unit": "iter/sec",
            "range": "stddev: 0.01389275859280193",
            "extra": "mean: 795.6228928000087 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.425738293672138,
            "unit": "iter/sec",
            "range": "stddev: 0.060917175575213166",
            "extra": "mean: 291.9078791999823 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.045770588369551,
            "unit": "iter/sec",
            "range": "stddev: 0.01983406137667159",
            "extra": "mean: 488.8133624000261 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8236579347012523,
            "unit": "iter/sec",
            "range": "stddev: 0.00943189364966999",
            "extra": "mean: 548.3484490000137 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.611124580606059,
            "unit": "iter/sec",
            "range": "stddev: 0.010627947331834624",
            "extra": "mean: 276.9220439999799 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 214.80321801636018,
            "unit": "iter/sec",
            "range": "stddev: 0.0000984439488488414",
            "extra": "mean: 4.655423737291666 msec\nrounds: 118"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3300055505627013,
            "unit": "iter/sec",
            "range": "stddev: 0.1509880278271563",
            "extra": "mean: 751.8765614000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6143546898352135,
            "unit": "iter/sec",
            "range": "stddev: 0.028816278168650965",
            "extra": "mean: 276.67456180001864 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "50394665+JunnYu@users.noreply.github.com",
            "name": "yujun",
            "username": "JunnYu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "443152565fe803138d0575b77c8e8ad651e985db",
          "message": "add paddle docs (#177)",
          "timestamp": "2023-02-08T11:06:05+01:00",
          "tree_id": "68621463baa6e29ad4140bcf9232b2b3ed33f037",
          "url": "https://github.com/huggingface/safetensors/commit/443152565fe803138d0575b77c8e8ad651e985db"
        },
        "date": 1675851098392,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2161499847537747,
            "unit": "iter/sec",
            "range": "stddev: 0.026968081311688197",
            "extra": "mean: 822.2670004000065 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.209697983827887,
            "unit": "iter/sec",
            "range": "stddev: 0.07616052418071073",
            "extra": "mean: 311.55579279998165 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.0784739625345363,
            "unit": "iter/sec",
            "range": "stddev: 0.017287482657579765",
            "extra": "mean: 481.1222165999993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8600712829537522,
            "unit": "iter/sec",
            "range": "stddev: 0.009996085869015747",
            "extra": "mean: 537.6138050000009 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7651973361083386,
            "unit": "iter/sec",
            "range": "stddev: 0.009380757430410417",
            "extra": "mean: 265.5903292000062 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 219.69444720522222,
            "unit": "iter/sec",
            "range": "stddev: 0.00019500498970133966",
            "extra": "mean: 4.5517764000010175 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.421322632788773,
            "unit": "iter/sec",
            "range": "stddev: 0.08858038535444458",
            "extra": "mean: 703.5700248000012 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.5577993283486506,
            "unit": "iter/sec",
            "range": "stddev: 0.052711528936954244",
            "extra": "mean: 281.0726260000024 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0c5b3a6b9c9be653b91ce8355225b8cad074f8b0",
          "message": "[Major Change] Enforcing tensor alignment (#148)\n\n* [Major Change] Enforcing tensor alignment\r\n\r\n- Now the header will automatically align itself to 8 bytes (f64) with\r\n  appending extra spaces as necessary.\r\n- This will allow extra fast memory mapping by reinterpreting bytes as\r\n  f32/f64 etc.. Unaligned bytes do not allow for this. https://www.reddit.com/r/rust/comments/tanaxm/mutating_a_buffer_of_u8s_as_f32s_in_place/\r\n- This does not change contiguousness of tensors\r\n- This does not change the actual spec (we're just putting extra valid bytes\r\n  in the header and using a different serialization ordering)\r\n- Readers should still be able to read old files, they would just need\r\n  to be copied before being cast as their final destination when using\r\n  mmap\r\n- This has no effect for GPU since copy is already necessary (*I think*,\r\n  depends on the cuda API actually if it allows filling f32 addresses\r\n  from raw unaligned bytes).\r\n\r\nThis change will only be interesting if things like https://github.com/Narsil/fast_gpt2\r\nactually pick up. And even with the copy, load times are still vastly\r\nsuperior to `pytorch`.\r\n\r\nWe need to be able to read old files.\r\n\r\n* Fixup.\r\n\r\n* Clippy fix.\r\n\r\n* Cargo fmt (clippy --fix broke it ? :( )",
          "timestamp": "2023-02-21T15:23:56+01:00",
          "tree_id": "4ea554c9d19f3f89f35d5072545a10e2c4c4356a",
          "url": "https://github.com/huggingface/safetensors/commit/0c5b3a6b9c9be653b91ce8355225b8cad074f8b0"
        },
        "date": 1676989854869,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0880325300906046,
            "unit": "iter/sec",
            "range": "stddev: 0.027575222551317807",
            "extra": "mean: 919.0901671999882 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.045821197644565,
            "unit": "iter/sec",
            "range": "stddev: 0.0728577698029528",
            "extra": "mean: 328.3186816000011 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.8361150631998868,
            "unit": "iter/sec",
            "range": "stddev: 0.022123201682182805",
            "extra": "mean: 544.6281771999907 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.6059027862952038,
            "unit": "iter/sec",
            "range": "stddev: 0.011073198802285416",
            "extra": "mean: 622.7026993999971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.179166400684258,
            "unit": "iter/sec",
            "range": "stddev: 0.01658308981899354",
            "extra": "mean: 314.54786380000996 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 98.10854765020046,
            "unit": "iter/sec",
            "range": "stddev: 0.00026469830903379547",
            "extra": "mean: 10.192791800011491 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2168850699580498,
            "unit": "iter/sec",
            "range": "stddev: 0.11252141101178768",
            "extra": "mean: 821.7702926000015 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.5049990553749684,
            "unit": "iter/sec",
            "range": "stddev: 0.012751864052489511",
            "extra": "mean: 285.30678159997933 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "fdwr@hotmail.com",
            "name": "Dwayne Robinson",
            "username": "fdwr"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "120c4da5128ee5730f66e972a058a5dcd20fa358",
          "message": "README.md proposed clarity (#181)\n\n* README.md proposed clarity\r\n\r\nI had to look via a hex editor at an existing .safetensors file to figure it out (maybe this will accelerate others slightly to adopt it 😅), because parentheses \"(...)\" are not legal JSON, and it wasn't clear to me whether the offsets were file relative or byte buffer relative.\r\n\r\n* Update other readme.\r\n\r\n---------\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2023-02-24T11:12:57+01:00",
          "tree_id": "ad78f432b64e24e4f939ccfb5e574e9b2255d7cf",
          "url": "https://github.com/huggingface/safetensors/commit/120c4da5128ee5730f66e972a058a5dcd20fa358"
        },
        "date": 1677233921694,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1935044565462292,
            "unit": "iter/sec",
            "range": "stddev: 0.04919540841759859",
            "extra": "mean: 837.8686770000058 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.172753268075411,
            "unit": "iter/sec",
            "range": "stddev: 0.07223332719511172",
            "extra": "mean: 315.1836640000056 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.0213827535629125,
            "unit": "iter/sec",
            "range": "stddev: 0.017590214596932097",
            "extra": "mean: 494.7108598 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8809788815891215,
            "unit": "iter/sec",
            "range": "stddev: 0.010412294273531587",
            "extra": "mean: 531.6380793999997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.806925511074714,
            "unit": "iter/sec",
            "range": "stddev: 0.009920869257562222",
            "extra": "mean: 262.67916120000336 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.82199978699423,
            "unit": "iter/sec",
            "range": "stddev: 0.00022754525358795126",
            "extra": "mean: 7.823379399997065 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.671979170927584,
            "unit": "iter/sec",
            "range": "stddev: 0.09355006132925571",
            "extra": "mean: 598.0935751999937 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.016255653181784,
            "unit": "iter/sec",
            "range": "stddev: 0.022384494744178075",
            "extra": "mean: 331.53688379999267 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "960cd17702430a2f033fa4dc5361119a6dac8b55",
          "message": "Better Rust API (tch-rs) (#179)\n\n* Implementing a trait API for the Rust (should make tch-rs interaction\r\neasier).\r\n\r\n* Companion type.\r\n\r\n* Re-export major formats + 3rd example in the trait doc.\r\n\r\n* Even clearer OpaqueGPU type.\r\n\r\n* Moving to `Cow<[u8]>`. Feels simpler.\r\n\r\n* Fixes.",
          "timestamp": "2023-02-24T14:56:51+01:00",
          "tree_id": "55feb684b8f87a659099c990c3660e4c04c113c4",
          "url": "https://github.com/huggingface/safetensors/commit/960cd17702430a2f033fa4dc5361119a6dac8b55"
        },
        "date": 1677247394544,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0347768715381531,
            "unit": "iter/sec",
            "range": "stddev: 0.007376250504844249",
            "extra": "mean: 966.3919125999996 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6542077982746073,
            "unit": "iter/sec",
            "range": "stddev: 0.08267021704527966",
            "extra": "mean: 376.760252400004 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7191263493138516,
            "unit": "iter/sec",
            "range": "stddev: 0.027732195687078513",
            "extra": "mean: 581.6908107999893 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.543859894926997,
            "unit": "iter/sec",
            "range": "stddev: 0.01919518388100529",
            "extra": "mean: 647.7271695999889 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.9246202486917094,
            "unit": "iter/sec",
            "range": "stddev: 0.014880281278605603",
            "extra": "mean: 341.9247337999991 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 71.77757141250999,
            "unit": "iter/sec",
            "range": "stddev: 0.0005970634714562729",
            "extra": "mean: 13.93192859999317 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.117735911740744,
            "unit": "iter/sec",
            "range": "stddev: 0.12867491021926386",
            "extra": "mean: 894.6657162000065 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.2014823857185393,
            "unit": "iter/sec",
            "range": "stddev: 0.006539923847096835",
            "extra": "mean: 312.35530280000603 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "488d945cc55ae48af70e13b5a351226ef5cc7758",
          "message": "Faster GPU Loads ?? (#184)\n\n* Faster GPU Loads ??\r\n\r\nBeats my expectations. I wonder what I did do to not see this perf\r\nbottleneck/improvements.\r\n\r\n* Pyslice still not fixed on PyO3.\r\n\r\n* Upgrade PyO3.\r\n\r\n* Using newer pyo3.\r\n\r\n* Make benches Windows friendly.",
          "timestamp": "2023-02-28T16:19:40+01:00",
          "tree_id": "1e9d436c693d720cb052c0bf22073d844ee0dc43",
          "url": "https://github.com/huggingface/safetensors/commit/488d945cc55ae48af70e13b5a351226ef5cc7758"
        },
        "date": 1677597938459,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1717156701031601,
            "unit": "iter/sec",
            "range": "stddev: 0.01793298919979068",
            "extra": "mean: 853.449369600014 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.121841519767719,
            "unit": "iter/sec",
            "range": "stddev: 0.07016782915511098",
            "extra": "mean: 320.32375559999764 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9061937249318666,
            "unit": "iter/sec",
            "range": "stddev: 0.015215503277082917",
            "extra": "mean: 524.6056510000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.7952959945905926,
            "unit": "iter/sec",
            "range": "stddev: 0.009951468286315417",
            "extra": "mean: 557.0112131999963 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.157883925591575,
            "unit": "iter/sec",
            "range": "stddev: 0.03769848981263098",
            "extra": "mean: 193.8779574000023 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 126.93718843279679,
            "unit": "iter/sec",
            "range": "stddev: 0.00021989960550117336",
            "extra": "mean: 7.877911999992193 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2578404667934968,
            "unit": "iter/sec",
            "range": "stddev: 0.06474215387057632",
            "extra": "mean: 795.0133791999974 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7017190804877242,
            "unit": "iter/sec",
            "range": "stddev: 0.0191132427524157",
            "extra": "mean: 270.14475660001835 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "47423c2294ea59930a56c204164fe0ec9fd944de",
          "message": "Preparing for release 0.3rc1. (#185)\n\n* Preparing for release 0.3rc1.\r\n\r\n* Full version number.",
          "timestamp": "2023-03-01T21:45:08+01:00",
          "tree_id": "d60446ab0a8bd49cee7905805f598ff8b9530918",
          "url": "https://github.com/huggingface/safetensors/commit/47423c2294ea59930a56c204164fe0ec9fd944de"
        },
        "date": 1677703859939,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4792282023071701,
            "unit": "iter/sec",
            "range": "stddev: 0.013846385002514798",
            "extra": "mean: 676.0282142000051 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.224819807991956,
            "unit": "iter/sec",
            "range": "stddev: 0.055952288118869806",
            "extra": "mean: 236.69648540000026 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7619158892167364,
            "unit": "iter/sec",
            "range": "stddev: 0.017220145994583227",
            "extra": "mean: 362.06750679999686 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.160351053693257,
            "unit": "iter/sec",
            "range": "stddev: 0.0105564388364372",
            "extra": "mean: 462.8877322000221 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.691422994297149,
            "unit": "iter/sec",
            "range": "stddev: 0.010508810568082817",
            "extra": "mean: 213.15494279999712 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 128.6612057876297,
            "unit": "iter/sec",
            "range": "stddev: 0.00016563308290490876",
            "extra": "mean: 7.77235059999839 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.93032271755213,
            "unit": "iter/sec",
            "range": "stddev: 0.030895073494206858",
            "extra": "mean: 518.0480915999965 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.7308605978433205,
            "unit": "iter/sec",
            "range": "stddev: 0.023633203153242636",
            "extra": "mean: 211.37803140001097 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b9b46822aa86b17185f9439b7023381678ecaa60",
          "message": "Version 0.3.0 (#186)",
          "timestamp": "2023-03-03T15:21:07+01:00",
          "tree_id": "e65ac65029a51986fdde68913e4a01cac8512d7a",
          "url": "https://github.com/huggingface/safetensors/commit/b9b46822aa86b17185f9439b7023381678ecaa60"
        },
        "date": 1677853623390,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3907902881284597,
            "unit": "iter/sec",
            "range": "stddev: 0.03262201593773734",
            "extra": "mean: 719.0156621999904 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.917247372082958,
            "unit": "iter/sec",
            "range": "stddev: 0.06776170892844523",
            "extra": "mean: 255.28129959999436 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.819016779583864,
            "unit": "iter/sec",
            "range": "stddev: 0.02044725799079926",
            "extra": "mean: 354.73361040001237 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.232833815541444,
            "unit": "iter/sec",
            "range": "stddev: 0.009188872131702774",
            "extra": "mean: 447.8613648000078 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.9054437744340555,
            "unit": "iter/sec",
            "range": "stddev: 0.009764918809957992",
            "extra": "mean: 203.85515480001004 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.1903578719597,
            "unit": "iter/sec",
            "range": "stddev: 0.0001788448909779862",
            "extra": "mean: 7.862231199999314 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.251150259442556,
            "unit": "iter/sec",
            "range": "stddev: 0.05115034528950131",
            "extra": "mean: 444.21734879999804 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.048894346111008,
            "unit": "iter/sec",
            "range": "stddev: 0.015540162965446207",
            "extra": "mean: 198.06316619999507 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5b3f1e298e080662c2d920713903156deedc454e",
          "message": "Remove too many keywords. (#187)",
          "timestamp": "2023-03-08T13:01:11+01:00",
          "tree_id": "f292e19e66f246be956a21cf31d7c67ed85483b0",
          "url": "https://github.com/huggingface/safetensors/commit/5b3f1e298e080662c2d920713903156deedc454e"
        },
        "date": 1678277223203,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.362351638298996,
            "unit": "iter/sec",
            "range": "stddev: 0.01221447438562372",
            "extra": "mean: 734.0248816000098 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.5112349983432125,
            "unit": "iter/sec",
            "range": "stddev: 0.08830876479552083",
            "extra": "mean: 284.8000776000049 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.703485120523412,
            "unit": "iter/sec",
            "range": "stddev: 0.016284586802688173",
            "extra": "mean: 369.892918000005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.16165655700318,
            "unit": "iter/sec",
            "range": "stddev: 0.010710743236513443",
            "extra": "mean: 462.60817739999993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.370854496381023,
            "unit": "iter/sec",
            "range": "stddev: 0.012286467790410587",
            "extra": "mean: 228.7882153999817 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.60281249984794,
            "unit": "iter/sec",
            "range": "stddev: 0.0002063002799558374",
            "extra": "mean: 7.59862179997981 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7637587634021294,
            "unit": "iter/sec",
            "range": "stddev: 0.12039081164998888",
            "extra": "mean: 566.9709603999877 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.600946557704318,
            "unit": "iter/sec",
            "range": "stddev: 0.015433753052521964",
            "extra": "mean: 217.34658020000097 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e275bad7006a2ace62a1fbc1421ec95959c3b880",
          "message": "Adding memmap example to read from file. (#189)",
          "timestamp": "2023-03-08T13:01:26+01:00",
          "tree_id": "449c17f759c1f776894fb0e8814bc78cee1d953d",
          "url": "https://github.com/huggingface/safetensors/commit/e275bad7006a2ace62a1fbc1421ec95959c3b880"
        },
        "date": 1678277237078,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3782384019458256,
            "unit": "iter/sec",
            "range": "stddev: 0.014122148329687155",
            "extra": "mean: 725.5638782000119 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.2121379322530053,
            "unit": "iter/sec",
            "range": "stddev: 0.11221121683915426",
            "extra": "mean: 311.319134200005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.8359978757581303,
            "unit": "iter/sec",
            "range": "stddev: 0.018850857802283332",
            "extra": "mean: 352.60957300000655 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.176628430030905,
            "unit": "iter/sec",
            "range": "stddev: 0.009467439859786823",
            "extra": "mean: 459.4261410000058 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.462941563951302,
            "unit": "iter/sec",
            "range": "stddev: 0.012057670669306308",
            "extra": "mean: 224.0674644000137 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 129.26558661826587,
            "unit": "iter/sec",
            "range": "stddev: 0.00020517927319492937",
            "extra": "mean: 7.736010999997234 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.078099871726761,
            "unit": "iter/sec",
            "range": "stddev: 0.08349950029003775",
            "extra": "mean: 481.2088261999975 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.724793556502309,
            "unit": "iter/sec",
            "range": "stddev: 0.01866962755480613",
            "extra": "mean: 174.67878799999426 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e32f5083dd778737ab65025e06fdf190955c841b",
          "message": "Making sure the safetensors existing PR is up-to-date. (#192)\n\n* Making sure the safetensors existing PR is up-to-date.\r\n\r\n* Fixed `target_branch` exists only on details.",
          "timestamp": "2023-03-13T11:35:27+01:00",
          "tree_id": "2ef97b4b4d1c8f51e1de0bf88d256c9acd80a779",
          "url": "https://github.com/huggingface/safetensors/commit/e32f5083dd778737ab65025e06fdf190955c841b"
        },
        "date": 1678704090667,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4466497709777892,
            "unit": "iter/sec",
            "range": "stddev: 0.010225127346524419",
            "extra": "mean: 691.2523128000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.9617862578599112,
            "unit": "iter/sec",
            "range": "stddev: 0.06529084043086357",
            "extra": "mean: 252.41139599999084 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.744240995534016,
            "unit": "iter/sec",
            "range": "stddev: 0.017409386070039425",
            "extra": "mean: 364.3994829999997 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1002202117016666,
            "unit": "iter/sec",
            "range": "stddev: 0.010601713690231935",
            "extra": "mean: 476.14054679997935 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.903891653190197,
            "unit": "iter/sec",
            "range": "stddev: 0.011707126689137768",
            "extra": "mean: 203.91967660000319 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 125.58216753285919,
            "unit": "iter/sec",
            "range": "stddev: 0.00048625148741564494",
            "extra": "mean: 7.962914000017918 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8898316537093234,
            "unit": "iter/sec",
            "range": "stddev: 0.082208166690782",
            "extra": "mean: 529.1476614000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.591367831632782,
            "unit": "iter/sec",
            "range": "stddev: 0.032962877208004164",
            "extra": "mean: 178.84711400000697 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "454924be76cb1c3c8c270dd775fc4c8280ae3ebd",
          "message": "Nicer error when failed to find tensor in Rust. (#191)\n\nCo-authored-by: Ubuntu <ubuntu@ip-172-31-34-94.eu-west-3.compute.internal>",
          "timestamp": "2023-03-13T11:35:40+01:00",
          "tree_id": "b753a0935b50e6b514d711435bff39a80c03b152",
          "url": "https://github.com/huggingface/safetensors/commit/454924be76cb1c3c8c270dd775fc4c8280ae3ebd"
        },
        "date": 1678704111641,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4490670116837148,
            "unit": "iter/sec",
            "range": "stddev: 0.017338337719444163",
            "extra": "mean: 690.0992100000053 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.786336610108969,
            "unit": "iter/sec",
            "range": "stddev: 0.07184989769850851",
            "extra": "mean: 264.1075274000059 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7248986637707553,
            "unit": "iter/sec",
            "range": "stddev: 0.0225253294435177",
            "extra": "mean: 366.9861244000117 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1118246495429878,
            "unit": "iter/sec",
            "range": "stddev: 0.01045600992111197",
            "extra": "mean: 473.5241632000111 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.776424674409221,
            "unit": "iter/sec",
            "range": "stddev: 0.010769971131204406",
            "extra": "mean: 209.3616183999984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 132.13460816781665,
            "unit": "iter/sec",
            "range": "stddev: 0.00018665785168716368",
            "extra": "mean: 7.568040000012388 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0019505051819344,
            "unit": "iter/sec",
            "range": "stddev: 0.05826375745625648",
            "extra": "mean: 499.5128487999864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.474659296957983,
            "unit": "iter/sec",
            "range": "stddev: 0.0494718031109325",
            "extra": "mean: 223.48070179998558 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7488c1dace4d66c46404f90fc960f0c112d48e90",
          "message": "Adding commit description to convertion PR. (#198)\n\n* Adding commit description to convertion PR.\r\n\r\n* Update bindings/python/convert.py\r\n\r\nCo-authored-by: Omar Sanseviero <osanseviero@gmail.com>\r\n\r\n* Adding suggestion for reporting issues.\r\n\r\n---------\r\n\r\nCo-authored-by: Omar Sanseviero <osanseviero@gmail.com>",
          "timestamp": "2023-03-17T10:42:34+01:00",
          "tree_id": "122389d6cdc3ddbb495fd83b4d405af933cbcc35",
          "url": "https://github.com/huggingface/safetensors/commit/7488c1dace4d66c46404f90fc960f0c112d48e90"
        },
        "date": 1679046613406,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.9968782283764324,
            "unit": "iter/sec",
            "range": "stddev: 0.03269275984958283",
            "extra": "mean: 1.0031315476000031 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.645688531018759,
            "unit": "iter/sec",
            "range": "stddev: 0.08012695602204807",
            "extra": "mean: 377.9734418000203 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.6791981970467718,
            "unit": "iter/sec",
            "range": "stddev: 0.02455539006171132",
            "extra": "mean: 595.5223164000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.536879182787286,
            "unit": "iter/sec",
            "range": "stddev: 0.011057098209039183",
            "extra": "mean: 650.6692336000015 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.883756834748849,
            "unit": "iter/sec",
            "range": "stddev: 0.010223616716361874",
            "extra": "mean: 346.7698759999962 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 73.88401046723757,
            "unit": "iter/sec",
            "range": "stddev: 0.0005700337551865725",
            "extra": "mean: 13.53472819999979 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.0965583497053262,
            "unit": "iter/sec",
            "range": "stddev: 0.13066130782418442",
            "extra": "mean: 911.9441753999922 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.4996726841098607,
            "unit": "iter/sec",
            "range": "stddev: 0.02803149294636537",
            "extra": "mean: 400.0523774000044 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c1d366813e46c6f9f2c71aa8b89e0c916a92b2f",
          "message": "Python 2.0 added a warning on ByteStorage, so using UntypedStorage (#196)\n\n* Python 2.0 added a warning on ByteStorage, so using UntypedStorage\r\n\r\ndirectly for that version.\r\n\r\nSmall error in their docs, `size` got renamed `nbytes`.\r\n\r\n* Remove unused work.",
          "timestamp": "2023-03-17T15:56:06+01:00",
          "tree_id": "4aa95791bb3bb59b85d3d9a413e76b3e56bdce4f",
          "url": "https://github.com/huggingface/safetensors/commit/5c1d366813e46c6f9f2c71aa8b89e0c916a92b2f"
        },
        "date": 1679065341061,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.400383545895966,
            "unit": "iter/sec",
            "range": "stddev: 0.013773270659666922",
            "extra": "mean: 714.0900812000041 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.7482269565247375,
            "unit": "iter/sec",
            "range": "stddev: 0.07838767554610884",
            "extra": "mean: 266.7928093999876 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.791593928243963,
            "unit": "iter/sec",
            "range": "stddev: 0.01738755399406334",
            "extra": "mean: 358.218288799992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1750434570636847,
            "unit": "iter/sec",
            "range": "stddev: 0.010156308481155545",
            "extra": "mean: 459.76092880001715 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.965503988346744,
            "unit": "iter/sec",
            "range": "stddev: 0.04935867930623112",
            "extra": "mean: 201.38942639998731 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.93416208692736,
            "unit": "iter/sec",
            "range": "stddev: 0.00016200664259860846",
            "extra": "mean: 7.637426200017217 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.072763108585412,
            "unit": "iter/sec",
            "range": "stddev: 0.05190212820065316",
            "extra": "mean: 482.4477992000084 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.116191996090944,
            "unit": "iter/sec",
            "range": "stddev: 0.00926254566557272",
            "extra": "mean: 242.94299219999402 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "19679563a1a55d74a07197045ec9d9d4c2698868",
          "message": "Adding a minimal version for torch. (#204)",
          "timestamp": "2023-03-27T10:30:56+02:00",
          "tree_id": "8e93bf1c04df4fa21094b83c4cee7767bcc596ee",
          "url": "https://github.com/huggingface/safetensors/commit/19679563a1a55d74a07197045ec9d9d4c2698868"
        },
        "date": 1679906253927,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.447621856744381,
            "unit": "iter/sec",
            "range": "stddev: 0.01699978829849029",
            "extra": "mean: 690.7881332000215 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.8119855709270496,
            "unit": "iter/sec",
            "range": "stddev: 0.07437963957171675",
            "extra": "mean: 262.33047880000413 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7236394694727943,
            "unit": "iter/sec",
            "range": "stddev: 0.017473782886771434",
            "extra": "mean: 367.15578959999675 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.143788296384786,
            "unit": "iter/sec",
            "range": "stddev: 0.007960761571780547",
            "extra": "mean: 466.46397020002723 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.5439393647420365,
            "unit": "iter/sec",
            "range": "stddev: 0.0112201773017238",
            "extra": "mean: 220.07335920002333 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 132.5505850146108,
            "unit": "iter/sec",
            "range": "stddev: 0.00014332288696552634",
            "extra": "mean: 7.544289600002685 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.902675154303154,
            "unit": "iter/sec",
            "range": "stddev: 0.06700210160990633",
            "extra": "mean: 525.5757914000014 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.510680735272606,
            "unit": "iter/sec",
            "range": "stddev: 0.020875716340514418",
            "extra": "mean: 181.4657840000109 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nickkolok@mail.ru",
            "name": "nickkolok",
            "username": "nickkolok"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a3f178ae840ee63c84f8c04c9a6622174afcf741",
          "message": "Minor improvements for README (#201)\n\n* Use `s for code\r\n\r\n* Fix typos in README\r\n\r\n* All readmes.\r\n\r\n* Adding Rust cache and latest ubuntu for python tests.\r\n\r\n---------\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2023-03-28T10:20:42+02:00",
          "tree_id": "666ce62d8b2e5e2b0bb583d2b8c361bfb6b076b7",
          "url": "https://github.com/huggingface/safetensors/commit/a3f178ae840ee63c84f8c04c9a6622174afcf741"
        },
        "date": 1679992035460,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2436472199258686,
            "unit": "iter/sec",
            "range": "stddev: 0.017399423395646905",
            "extra": "mean: 804.0865480000093 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.4096958588800015,
            "unit": "iter/sec",
            "range": "stddev: 0.06421916087185138",
            "extra": "mean: 293.28129000000445 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.1182055059122638,
            "unit": "iter/sec",
            "range": "stddev: 0.020132660349338763",
            "extra": "mean: 472.097724799994 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8525374505756262,
            "unit": "iter/sec",
            "range": "stddev: 0.010303374997704964",
            "extra": "mean: 539.8001534000173 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.702827595494817,
            "unit": "iter/sec",
            "range": "stddev: 0.009929890957780888",
            "extra": "mean: 270.06388339999603 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 125.73616165101376,
            "unit": "iter/sec",
            "range": "stddev: 0.0004144179540179656",
            "extra": "mean: 7.953161499995077 msec\nrounds: 10"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.67586017077762,
            "unit": "iter/sec",
            "range": "stddev: 0.08388227764623506",
            "extra": "mean: 596.7084948000092 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.8625297660641182,
            "unit": "iter/sec",
            "range": "stddev: 0.02832503267765318",
            "extra": "mean: 258.89768120000554 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b39a2fe8f6f8d484880e52a37fdb1ef199b1de94",
          "message": "Remove unwrap in prod code. (#209)",
          "timestamp": "2023-03-28T12:26:19+02:00",
          "tree_id": "eb14413869ae38f324711ce1654b69c55d0e59c0",
          "url": "https://github.com/huggingface/safetensors/commit/b39a2fe8f6f8d484880e52a37fdb1ef199b1de94"
        },
        "date": 1679999566245,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4777256680906128,
            "unit": "iter/sec",
            "range": "stddev: 0.040417940540626585",
            "extra": "mean: 676.7155918000071 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.052821278926472,
            "unit": "iter/sec",
            "range": "stddev: 0.06472236824075626",
            "extra": "mean: 246.7416969999931 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7975308446686222,
            "unit": "iter/sec",
            "range": "stddev: 0.017267928062230035",
            "extra": "mean: 357.4580784000091 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.2063597770747325,
            "unit": "iter/sec",
            "range": "stddev: 0.010165626070027047",
            "extra": "mean: 453.23523860004116 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.244864113652082,
            "unit": "iter/sec",
            "range": "stddev: 0.01012737777689595",
            "extra": "mean: 235.5788013999927 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 134.2279296215407,
            "unit": "iter/sec",
            "range": "stddev: 0.00003983204024755645",
            "extra": "mean: 7.450014336208024 msec\nrounds: 116"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.770027865743392,
            "unit": "iter/sec",
            "range": "stddev: 0.0720944299915913",
            "extra": "mean: 564.9628570000004 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.29552918453034,
            "unit": "iter/sec",
            "range": "stddev: 0.0071861187830230665",
            "extra": "mean: 232.8001876000144 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ba5ed42013185b8542901998d93b3b77c2e1196b",
          "message": "More badges. (#210)\n\n* More badges.\r\n\r\n* Update action to create codecov.\r\n\r\n* Update readmes.\r\n\r\n* Fix doc link.\r\n\r\n* Two lines for Python and Rust.\r\n\r\n* Doc link.\r\n\r\n* Fix badge label.\r\n\r\n* Use simpler run to get default working-dir.\r\n\r\n* Fix ?\r\n\r\n* Audit.\r\n\r\n* Ignore criterion issue.\r\n\r\n* Better command line ?\r\n\r\n* codecov token.\r\n\r\n* Working dir.\r\n\r\n* Specify file.\r\n\r\n* Fixing upload only on ubuntu.\r\n\r\n* Code coverage only on linux.\r\n\r\n* Fix codecov link.",
          "timestamp": "2023-03-28T12:24:44+02:00",
          "tree_id": "2727ae5497c0c8dce19a4561b1c952848a93c5ef",
          "url": "https://github.com/huggingface/safetensors/commit/ba5ed42013185b8542901998d93b3b77c2e1196b"
        },
        "date": 1679999582016,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1434870785192026,
            "unit": "iter/sec",
            "range": "stddev: 0.035284906032149725",
            "extra": "mean: 874.5179711999754 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.837245479950206,
            "unit": "iter/sec",
            "range": "stddev: 0.06783637320022129",
            "extra": "mean: 352.45452220001425 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.8851239722843398,
            "unit": "iter/sec",
            "range": "stddev: 0.02103772474974068",
            "extra": "mean: 530.469091000009 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.644242958035541,
            "unit": "iter/sec",
            "range": "stddev: 0.012321207975665592",
            "extra": "mean: 608.1826259999616 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.098210979895401,
            "unit": "iter/sec",
            "range": "stddev: 0.009477282900591643",
            "extra": "mean: 322.76691500001107 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 62.21215712466327,
            "unit": "iter/sec",
            "range": "stddev: 0.001570442803684191",
            "extra": "mean: 16.07402871429388 msec\nrounds: 7"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3876848567990454,
            "unit": "iter/sec",
            "range": "stddev: 0.11329632998394429",
            "extra": "mean: 720.6247117999737 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.5965573761424174,
            "unit": "iter/sec",
            "range": "stddev: 0.017996640387714164",
            "extra": "mean: 278.04366660002415 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e619689636eb640c974af6d040e4bf449e0eaebb",
          "message": "Upload codecov for main branch. (#211)",
          "timestamp": "2023-03-28T12:33:56+02:00",
          "tree_id": "e62fd26049fc133af680529e57791b1df8ab98f2",
          "url": "https://github.com/huggingface/safetensors/commit/e619689636eb640c974af6d040e4bf449e0eaebb"
        },
        "date": 1680000106106,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1635806631400252,
            "unit": "iter/sec",
            "range": "stddev: 0.03738825627938192",
            "extra": "mean: 859.4161382000038 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1108064188212086,
            "unit": "iter/sec",
            "range": "stddev: 0.07581869992890453",
            "extra": "mean: 321.46005420000847 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9797482782294435,
            "unit": "iter/sec",
            "range": "stddev: 0.019291219034875892",
            "extra": "mean: 505.11472140000245 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.906550269978016,
            "unit": "iter/sec",
            "range": "stddev: 0.009818396883339773",
            "extra": "mean: 524.5075441999916 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.546248909788361,
            "unit": "iter/sec",
            "range": "stddev: 0.010598950700910728",
            "extra": "mean: 281.98810219998904 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.45183881803422,
            "unit": "iter/sec",
            "range": "stddev: 0.0002101257368967815",
            "extra": "mean: 7.8461009999841735 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.645711226579796,
            "unit": "iter/sec",
            "range": "stddev: 0.06977671101205131",
            "extra": "mean: 607.6400183999795 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.104558603152102,
            "unit": "iter/sec",
            "range": "stddev: 0.014721904591726",
            "extra": "mean: 243.63155619998906 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f69601560a586cbdc1c896c75c2146ccde6a355",
          "message": "Oops. (#212)",
          "timestamp": "2023-03-28T12:40:20+02:00",
          "tree_id": "acb11e04bf6cf74102a5d34213c28ae365db9288",
          "url": "https://github.com/huggingface/safetensors/commit/3f69601560a586cbdc1c896c75c2146ccde6a355"
        },
        "date": 1680000398998,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.5211854100168432,
            "unit": "iter/sec",
            "range": "stddev: 0.02783714977670744",
            "extra": "mean: 657.3820609999984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.803093411689032,
            "unit": "iter/sec",
            "range": "stddev: 0.07558022487633824",
            "extra": "mean: 262.9438438000079 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.8010319638150256,
            "unit": "iter/sec",
            "range": "stddev: 0.019362882933435757",
            "extra": "mean: 357.011277599986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.3636732261947047,
            "unit": "iter/sec",
            "range": "stddev: 0.010359204146640242",
            "extra": "mean: 423.0703250000033 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.916574956319129,
            "unit": "iter/sec",
            "range": "stddev: 0.011270508180678915",
            "extra": "mean: 203.39362439999604 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.56141014713617,
            "unit": "iter/sec",
            "range": "stddev: 0.00020093473381351578",
            "extra": "mean: 7.659230999979627 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8490797430040065,
            "unit": "iter/sec",
            "range": "stddev: 0.059842553766232265",
            "extra": "mean: 540.8095587999924 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.46464942774884,
            "unit": "iter/sec",
            "range": "stddev: 0.017686267340879382",
            "extra": "mean: 223.98175179998816 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59686839393ed6c2e3cd554c1fa91806e808deda",
          "message": "Fixing codecov rust tag. (#213)",
          "timestamp": "2023-03-28T12:46:09+02:00",
          "tree_id": "1892347d367f38d10a3b7cb19e9edf1da5ff4dcf",
          "url": "https://github.com/huggingface/safetensors/commit/59686839393ed6c2e3cd554c1fa91806e808deda"
        },
        "date": 1680000762118,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.5015639250760853,
            "unit": "iter/sec",
            "range": "stddev: 0.03232464443883163",
            "extra": "mean: 665.9723128000223 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.677436153177182,
            "unit": "iter/sec",
            "range": "stddev: 0.07604656573389215",
            "extra": "mean: 271.9285824000053 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.684966127052598,
            "unit": "iter/sec",
            "range": "stddev: 0.01751249692642271",
            "extra": "mean: 372.4441772000091 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1283330143551438,
            "unit": "iter/sec",
            "range": "stddev: 0.011616084081646329",
            "extra": "mean: 469.85128419999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.14254256657827,
            "unit": "iter/sec",
            "range": "stddev: 0.010683823300240054",
            "extra": "mean: 241.39764019998893 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.82318103247212,
            "unit": "iter/sec",
            "range": "stddev: 0.0001694577862320837",
            "extra": "mean: 7.585919200005264 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7469917298557436,
            "unit": "iter/sec",
            "range": "stddev: 0.03145583088201071",
            "extra": "mean: 572.4125552000032 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.4521072082522277,
            "unit": "iter/sec",
            "range": "stddev: 0.0113233435658398",
            "extra": "mean: 289.6781413999861 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "73aba47ccff227f77e6fb7ef8f9d857f3b0a5676",
          "message": "Increase coverage on the error paths. (#214)",
          "timestamp": "2023-03-28T14:13:38+02:00",
          "tree_id": "7219b2dadda7fd4e22f8dbcf4bf3420c9bd89abe",
          "url": "https://github.com/huggingface/safetensors/commit/73aba47ccff227f77e6fb7ef8f9d857f3b0a5676"
        },
        "date": 1680006066086,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1665149931987282,
            "unit": "iter/sec",
            "range": "stddev: 0.029992129655972748",
            "extra": "mean: 857.2543052000356 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0368274165829003,
            "unit": "iter/sec",
            "range": "stddev: 0.07359879761216899",
            "extra": "mean: 329.29102079999666 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9566895632882395,
            "unit": "iter/sec",
            "range": "stddev: 0.029383775669240865",
            "extra": "mean: 511.0672734000218 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.6820568385872303,
            "unit": "iter/sec",
            "range": "stddev: 0.011661742709860244",
            "extra": "mean: 594.510231199979 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.068041986519749,
            "unit": "iter/sec",
            "range": "stddev: 0.016691010326153347",
            "extra": "mean: 325.94078060005813 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 115.18251614211674,
            "unit": "iter/sec",
            "range": "stddev: 0.0002096663984983567",
            "extra": "mean: 8.681873199975598 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4545972157791083,
            "unit": "iter/sec",
            "range": "stddev: 0.04758739268214718",
            "extra": "mean: 687.4755355999923 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.618425930602725,
            "unit": "iter/sec",
            "range": "stddev: 0.022525580362481232",
            "extra": "mean: 276.36326380002174 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4ea1fe533c2d5e9a4521234bed00ba27b3307e5a",
          "message": "Deactivating codecov commenting. (#217)",
          "timestamp": "2023-03-28T15:20:39+02:00",
          "tree_id": "994c6c391af952537bb245b9d7b9ea5cbfb85ab1",
          "url": "https://github.com/huggingface/safetensors/commit/4ea1fe533c2d5e9a4521234bed00ba27b3307e5a"
        },
        "date": 1680010037129,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4431184826175227,
            "unit": "iter/sec",
            "range": "stddev: 0.04153418384213563",
            "extra": "mean: 692.9437964000044 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9236402527794643,
            "unit": "iter/sec",
            "range": "stddev: 0.10602721243465313",
            "extra": "mean: 342.03934600001276 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6870242502648627,
            "unit": "iter/sec",
            "range": "stddev: 0.01823233755182608",
            "extra": "mean: 372.1589040000026 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.120134855892595,
            "unit": "iter/sec",
            "range": "stddev: 0.007892726551111496",
            "extra": "mean: 471.66810980002083 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 6.8175332693405215,
            "unit": "iter/sec",
            "range": "stddev: 0.028579976831625383",
            "extra": "mean: 146.68061900000566 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.29482658019869,
            "unit": "iter/sec",
            "range": "stddev: 0.00019262739738650284",
            "extra": "mean: 7.855778800012558 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7536441389839865,
            "unit": "iter/sec",
            "range": "stddev: 0.042736316637993106",
            "extra": "mean: 570.2411211999788 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.641864007550879,
            "unit": "iter/sec",
            "range": "stddev: 0.01110339103209852",
            "extra": "mean: 177.24638499999892 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8375de6d25673c78de9947bded483ca3359fea82",
          "message": "Zero width shape are OK (0-rank tensors). They are simply a scalar. (#208)\n\n* Zero width shape are OK (0-rank tensors). They are simply a scalar.\r\n\r\nReasosn for support is that both torch and numpy support them, and\r\nthey flow naturally from the definition that an empty product has\r\nlength 1.\r\n\r\n* not allowed -> allowed",
          "timestamp": "2023-03-28T15:32:41+02:00",
          "tree_id": "5d90363abe6164bfec88e0eab2788fa4c95fd3fa",
          "url": "https://github.com/huggingface/safetensors/commit/8375de6d25673c78de9947bded483ca3359fea82"
        },
        "date": 1680011093532,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2872687575924218,
            "unit": "iter/sec",
            "range": "stddev: 0.014502141046900374",
            "extra": "mean: 776.8385537999848 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3776600847284155,
            "unit": "iter/sec",
            "range": "stddev: 0.05610539139703431",
            "extra": "mean: 296.0629473999916 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.132778773542722,
            "unit": "iter/sec",
            "range": "stddev: 0.01736996801466173",
            "extra": "mean: 468.8718832000177 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.9770419702230366,
            "unit": "iter/sec",
            "range": "stddev: 0.013575482170632395",
            "extra": "mean: 505.8061564000014 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.6092245253388633,
            "unit": "iter/sec",
            "range": "stddev: 0.03456927267460218",
            "extra": "mean: 277.0678280000084 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 85.28969266727529,
            "unit": "iter/sec",
            "range": "stddev: 0.0011558255264619995",
            "extra": "mean: 11.7247462000023 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6147054741126459,
            "unit": "iter/sec",
            "range": "stddev: 0.0721822850410305",
            "extra": "mean: 619.3079890000035 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.3537049242547408,
            "unit": "iter/sec",
            "range": "stddev: 0.03603619920309045",
            "extra": "mean: 298.1776938000053 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e0b932d76da8328e4c0e0a5bb7d6a00fcc06aba0",
          "message": "Adding simple proptest. (#216)",
          "timestamp": "2023-03-28T15:45:08+02:00",
          "tree_id": "4a4ea5d1077e30a018f62f0501f5e08c5900a8cb",
          "url": "https://github.com/huggingface/safetensors/commit/e0b932d76da8328e4c0e0a5bb7d6a00fcc06aba0"
        },
        "date": 1680011579366,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3245969285059134,
            "unit": "iter/sec",
            "range": "stddev: 0.021002910691009943",
            "extra": "mean: 754.9466395999843 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.314133258020031,
            "unit": "iter/sec",
            "range": "stddev: 0.06774350216092394",
            "extra": "mean: 301.73801779999394 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.089281502100274,
            "unit": "iter/sec",
            "range": "stddev: 0.018771805271798173",
            "extra": "mean: 478.63344359998337 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8273526512655591,
            "unit": "iter/sec",
            "range": "stddev: 0.010280666526956627",
            "extra": "mean: 547.2397455999726 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.711316529363281,
            "unit": "iter/sec",
            "range": "stddev: 0.010378483695338534",
            "extra": "mean: 269.4461634000163 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 117.112403501313,
            "unit": "iter/sec",
            "range": "stddev: 0.00046096272047728643",
            "extra": "mean: 8.53880519998711 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6397127014382564,
            "unit": "iter/sec",
            "range": "stddev: 0.07258132035692959",
            "extra": "mean: 609.8629345999825 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7143396602935077,
            "unit": "iter/sec",
            "range": "stddev: 0.04128006124292803",
            "extra": "mean: 269.226859000014 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "446f7ade0b068c0c8b5e81c80490c2408b755fd7",
          "message": "Disallowing zero-width tensor. (#205)\n\n* Disallowing zero-width tensor.\r\n\r\n* Fix test.",
          "timestamp": "2023-03-28T15:45:39+02:00",
          "tree_id": "5db55819cb45eccb7f670549aeecb40f7ce65dd7",
          "url": "https://github.com/huggingface/safetensors/commit/446f7ade0b068c0c8b5e81c80490c2408b755fd7"
        },
        "date": 1680011629225,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.146110944230713,
            "unit": "iter/sec",
            "range": "stddev: 0.018439751635741318",
            "extra": "mean: 872.5158808000174 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9746445286402494,
            "unit": "iter/sec",
            "range": "stddev: 0.07908313330703473",
            "extra": "mean: 336.17462199999864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9592536794426647,
            "unit": "iter/sec",
            "range": "stddev: 0.021543156555455408",
            "extra": "mean: 510.3984289999971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8604867477690294,
            "unit": "iter/sec",
            "range": "stddev: 0.010313164066741414",
            "extra": "mean: 537.4937506000151 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.429190635368992,
            "unit": "iter/sec",
            "range": "stddev: 0.011876941654333312",
            "extra": "mean: 291.6140005999978 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 117.77343664860713,
            "unit": "iter/sec",
            "range": "stddev: 0.0005718219427377383",
            "extra": "mean: 8.490879000021323 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2988678564062806,
            "unit": "iter/sec",
            "range": "stddev: 0.05568438356197441",
            "extra": "mean: 769.9012605999883 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.5981440404589393,
            "unit": "iter/sec",
            "range": "stddev: 0.016389752971380577",
            "extra": "mean: 277.92105839999977 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7969ecce6effd87985fd3c3593d7c5e1265181d",
          "message": "Disallow erroneous buffer. (#206)",
          "timestamp": "2023-03-28T15:47:23+02:00",
          "tree_id": "f3198b64ee120b37018ca46705e555bf60feed20",
          "url": "https://github.com/huggingface/safetensors/commit/b7969ecce6effd87985fd3c3593d7c5e1265181d"
        },
        "date": 1680011682500,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4384123103647475,
            "unit": "iter/sec",
            "range": "stddev: 0.026892530722112507",
            "extra": "mean: 695.210957800009 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.639937162541778,
            "unit": "iter/sec",
            "range": "stddev: 0.08129512380191695",
            "extra": "mean: 274.73001739999745 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7868519923614676,
            "unit": "iter/sec",
            "range": "stddev: 0.017735644421850858",
            "extra": "mean: 358.82781099998056 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1138303109975554,
            "unit": "iter/sec",
            "range": "stddev: 0.014648915733118315",
            "extra": "mean: 473.0748702000028 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.694571532859483,
            "unit": "iter/sec",
            "range": "stddev: 0.012006670041486764",
            "extra": "mean: 213.01198480000494 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 134.23733699413677,
            "unit": "iter/sec",
            "range": "stddev: 0.00011201114360046695",
            "extra": "mean: 7.449492238092283 msec\nrounds: 21"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.9315675367857268,
            "unit": "iter/sec",
            "range": "stddev: 0.06737612454546239",
            "extra": "mean: 517.71422999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 6.505485917354378,
            "unit": "iter/sec",
            "range": "stddev: 0.0071365164996436735",
            "extra": "mean: 153.71641914285712 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "db4f5f4ad1aa2f920c30b2cfa836880a614e4e00",
          "message": "Overspecify a bit the format. (#215)\n\n* Updating the README to overspecify JSON.\r\n\r\n* Overspecify a bit the format.\r\n\r\n* Stop being annoying.\r\n\r\n* Fixing python cache + gh action.\r\n\r\n* Fix audit.\r\n\r\n* Fixing rust cache for python directory ?",
          "timestamp": "2023-03-28T15:53:50+02:00",
          "tree_id": "9a059727033d46d4cfeebba32c1739dbe6b0db7c",
          "url": "https://github.com/huggingface/safetensors/commit/db4f5f4ad1aa2f920c30b2cfa836880a614e4e00"
        },
        "date": 1680012100885,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2001083774671706,
            "unit": "iter/sec",
            "range": "stddev: 0.03386934609043146",
            "extra": "mean: 833.2580780000058 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0599314807779585,
            "unit": "iter/sec",
            "range": "stddev: 0.0794316566780917",
            "extra": "mean: 326.80470340001193 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.1177536179071645,
            "unit": "iter/sec",
            "range": "stddev: 0.01852752736912275",
            "extra": "mean: 472.198461399978 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8435865528358717,
            "unit": "iter/sec",
            "range": "stddev: 0.009758007788386034",
            "extra": "mean: 542.4209666000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7340827990009102,
            "unit": "iter/sec",
            "range": "stddev: 0.010517011342373959",
            "extra": "mean: 267.8033813999946 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 126.0678863968912,
            "unit": "iter/sec",
            "range": "stddev: 0.00016780483787380383",
            "extra": "mean: 7.93223420000686 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7379336936411942,
            "unit": "iter/sec",
            "range": "stddev: 0.02793086674843835",
            "extra": "mean: 575.3959449999911 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.2013373845592135,
            "unit": "iter/sec",
            "range": "stddev: 0.009994868608272948",
            "extra": "mean: 312.36945059999925 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3298667fa71c66ee57602b89368b89ec7c0330a2",
          "message": "Adding arithmetic overflow checks (unchecked in release builds). (#207)\n\n* Adding arithmetic overflow checks (unchecked in release builds).\r\n\r\n* Fix rebase",
          "timestamp": "2023-03-29T10:57:14+02:00",
          "tree_id": "ab554a05a03af8667c6d6ddac540bf15ed862789",
          "url": "https://github.com/huggingface/safetensors/commit/3298667fa71c66ee57602b89368b89ec7c0330a2"
        },
        "date": 1680080711970,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.064811765706605,
            "unit": "iter/sec",
            "range": "stddev: 0.044446444021574126",
            "extra": "mean: 939.133124 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.62233728697914,
            "unit": "iter/sec",
            "range": "stddev: 0.08123287126421806",
            "extra": "mean: 381.3391988000035 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7138702026144408,
            "unit": "iter/sec",
            "range": "stddev: 0.028775953734932008",
            "extra": "mean: 583.474756999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5728674837656662,
            "unit": "iter/sec",
            "range": "stddev: 0.014040827163706202",
            "extra": "mean: 635.7814693999899 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.9784835574938477,
            "unit": "iter/sec",
            "range": "stddev: 0.009862703900286167",
            "extra": "mean: 335.74131960003797 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 71.60516818600495,
            "unit": "iter/sec",
            "range": "stddev: 0.0008570644475540856",
            "extra": "mean: 13.965472400013823 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2182969774643793,
            "unit": "iter/sec",
            "range": "stddev: 0.1337419795434898",
            "extra": "mean: 820.8179274000031 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.192363739262753,
            "unit": "iter/sec",
            "range": "stddev: 0.03022055829734764",
            "extra": "mean: 313.24751239999387 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "91c7aa4e06f1b10d9769fae1d883c8f5b6a556a9",
          "message": "Fixing empty shape test. (#218)",
          "timestamp": "2023-03-29T10:58:08+02:00",
          "tree_id": "71f9c6dab4d6ebefdfa90b52996c8fde9e543904",
          "url": "https://github.com/huggingface/safetensors/commit/91c7aa4e06f1b10d9769fae1d883c8f5b6a556a9"
        },
        "date": 1680080715982,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2439769679843844,
            "unit": "iter/sec",
            "range": "stddev: 0.028164103522283972",
            "extra": "mean: 803.8734041999987 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.1619213684845424,
            "unit": "iter/sec",
            "range": "stddev: 0.06972778385102164",
            "extra": "mean: 316.26339919998827 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.010237649044417,
            "unit": "iter/sec",
            "range": "stddev: 0.01461589715469358",
            "extra": "mean: 497.4536222000211 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.7310841970780682,
            "unit": "iter/sec",
            "range": "stddev: 0.015462686744754303",
            "extra": "mean: 577.6726525999834 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.543148860466505,
            "unit": "iter/sec",
            "range": "stddev: 0.010117371515594647",
            "extra": "mean: 282.2348254000076 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 125.99660796929929,
            "unit": "iter/sec",
            "range": "stddev: 0.00048214665982951804",
            "extra": "mean: 7.936721600026431 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2876271221721907,
            "unit": "iter/sec",
            "range": "stddev: 0.05346774072554222",
            "extra": "mean: 776.6223488000378 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.0184500598698665,
            "unit": "iter/sec",
            "range": "stddev: 0.03304766624260868",
            "extra": "mean: 331.29585720000705 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ff6b7ba450e0349fbf54857d5203f607b5436699",
          "message": "Add a safety warning in the convertion script. (#219)\n\n* Add a safety warning in the convertion script.\r\n\r\n* Fix test temporarily.",
          "timestamp": "2023-03-29T10:57:32+02:00",
          "tree_id": "71f9c6dab4d6ebefdfa90b52996c8fde9e543904",
          "url": "https://github.com/huggingface/safetensors/commit/ff6b7ba450e0349fbf54857d5203f607b5436699"
        },
        "date": 1680080757985,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.9786306763335222,
            "unit": "iter/sec",
            "range": "stddev: 0.03746068801890575",
            "extra": "mean: 1.0218359430000077 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.562556736575736,
            "unit": "iter/sec",
            "range": "stddev: 0.07968880897009327",
            "extra": "mean: 390.23526219999667 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.6912313859408263,
            "unit": "iter/sec",
            "range": "stddev: 0.03194329486355372",
            "extra": "mean: 591.285147799988 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5167574643894757,
            "unit": "iter/sec",
            "range": "stddev: 0.01770455753682502",
            "extra": "mean: 659.3011892000277 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.815423674250844,
            "unit": "iter/sec",
            "range": "stddev: 0.00673285597594766",
            "extra": "mean: 355.18632920002347 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 66.2740477204267,
            "unit": "iter/sec",
            "range": "stddev: 0.0013059357750897793",
            "extra": "mean: 15.088862600009634 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.0905957189765587,
            "unit": "iter/sec",
            "range": "stddev: 0.100943379474908",
            "extra": "mean: 916.9300618000079 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.2580009598239585,
            "unit": "iter/sec",
            "range": "stddev: 0.0199529392310294",
            "extra": "mean: 306.9366806000062 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ec6fc4a174e822b5bbaed87b617bfabf9a2f71e5",
          "message": "Remove the breaking change of disallowing zero sized tensors. (#221)\n\n* Remove the breaking change of disallowing zero sized tensors.\r\n\r\nSome files out there might exist with this.\r\nWhile being questionnable, it's not worth doing a breaking change\r\nsince there doesn't seem to be either a security implication\r\nnor a performance hit.\r\n\r\n* Full support for zero-sized tensor on torch==1.10\r\n\r\n* Adding a nice comment.",
          "timestamp": "2023-03-30T14:58:08+02:00",
          "tree_id": "f08367be34c510cb0e566b998fc05238ddbc0786",
          "url": "https://github.com/huggingface/safetensors/commit/ec6fc4a174e822b5bbaed87b617bfabf9a2f71e5"
        },
        "date": 1680181535078,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0815306968931357,
            "unit": "iter/sec",
            "range": "stddev: 0.013906194432007333",
            "extra": "mean: 924.6154574000116 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.89173346383497,
            "unit": "iter/sec",
            "range": "stddev: 0.07129839507386429",
            "extra": "mean: 345.81333739998854 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7056401338018412,
            "unit": "iter/sec",
            "range": "stddev: 0.011742720578672839",
            "extra": "mean: 586.2901442000066 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.595083683093802,
            "unit": "iter/sec",
            "range": "stddev: 0.012799228205621429",
            "extra": "mean: 626.9263553999963 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.9738891727679873,
            "unit": "iter/sec",
            "range": "stddev: 0.008836012720205597",
            "extra": "mean: 336.26000900001145 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 73.27314433009975,
            "unit": "iter/sec",
            "range": "stddev: 0.0007873589949785551",
            "extra": "mean: 13.647565000007944 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3327298473146616,
            "unit": "iter/sec",
            "range": "stddev: 0.09630053823976574",
            "extra": "mean: 750.3396146 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.209765414949455,
            "unit": "iter/sec",
            "range": "stddev: 0.016435875486119647",
            "extra": "mean: 311.5492475999986 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72594fcce7cf4249ccb62dc5889a63a32dd586cc",
          "message": "0-sized tensors allowed. (#226)",
          "timestamp": "2023-04-06T10:27:30+02:00",
          "tree_id": "b10cf8d7edbd8a76340a701971e26c8b4c22fc31",
          "url": "https://github.com/huggingface/safetensors/commit/72594fcce7cf4249ccb62dc5889a63a32dd586cc"
        },
        "date": 1680770018595,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3543129233207762,
            "unit": "iter/sec",
            "range": "stddev: 0.01191466243905311",
            "extra": "mean: 738.3817895999982 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.369136733877566,
            "unit": "iter/sec",
            "range": "stddev: 0.09917072346029625",
            "extra": "mean: 296.811936999984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6975869666997037,
            "unit": "iter/sec",
            "range": "stddev: 0.017039025528075243",
            "extra": "mean: 370.70167239999137 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1338494658830407,
            "unit": "iter/sec",
            "range": "stddev: 0.009141789812995295",
            "extra": "mean: 468.63661940003567 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.855290555587853,
            "unit": "iter/sec",
            "range": "stddev: 0.008876175071278986",
            "extra": "mean: 259.38382220001586 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.67743169139277,
            "unit": "iter/sec",
            "range": "stddev: 0.0001807393895771475",
            "extra": "mean: 7.594315800020013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7154188444959413,
            "unit": "iter/sec",
            "range": "stddev: 0.1512362555731121",
            "extra": "mean: 582.9480090000061 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.6549674926710924,
            "unit": "iter/sec",
            "range": "stddev: 0.0784094839942535",
            "extra": "mean: 273.6002446000384 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca15ad0d5e51ecd5ce845bfa50b36a6c57c964e1",
          "message": "Adding last missing test for validate. (#225)",
          "timestamp": "2023-04-06T10:27:13+02:00",
          "tree_id": "7febe94501ec96ff877adc392d4573197a39eb55",
          "url": "https://github.com/huggingface/safetensors/commit/ca15ad0d5e51ecd5ce845bfa50b36a6c57c964e1"
        },
        "date": 1680770038837,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0499249537581647,
            "unit": "iter/sec",
            "range": "stddev: 0.017365146892671792",
            "extra": "mean: 952.4490263999724 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9231424168472246,
            "unit": "iter/sec",
            "range": "stddev: 0.08076370603788577",
            "extra": "mean: 342.09759820000727 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.8347282784666008,
            "unit": "iter/sec",
            "range": "stddev: 0.02613891446856877",
            "extra": "mean: 545.0398360000008 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.617970835202659,
            "unit": "iter/sec",
            "range": "stddev: 0.012591687881250974",
            "extra": "mean: 618.0581121999921 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.350255295993365,
            "unit": "iter/sec",
            "range": "stddev: 0.041611928661671536",
            "extra": "mean: 229.87156660001347 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 104.82522458868924,
            "unit": "iter/sec",
            "range": "stddev: 0.00016882134671678274",
            "extra": "mean: 9.53968859998895 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1932450932707923,
            "unit": "iter/sec",
            "range": "stddev: 0.09222865614084673",
            "extra": "mean: 838.0507957999725 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.5352288055328964,
            "unit": "iter/sec",
            "range": "stddev: 0.03679927999200205",
            "extra": "mean: 394.44171580000784 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0b21039218bd869969838d60286fe6786a87c9c3",
          "message": "Minor checks + proptest in python. (#228)\n\n* Minor checks + proptest in python.\r\n\r\n* Fix readme.\r\n\r\n* Tf version.\r\n\r\n* Fix.\r\n\r\n* Jax version.\r\n\r\n* Jaxlib.\r\n\r\n* Jax.\r\n\r\n* Cannot have jax version\r\n\r\n- Python 3.7 is minimal version for which they don't have release\r\n  jax>0.4\r\n- Our doc builder relies on jax>0.4\r\n\r\n* wtf.\r\n\r\n* .\r\n\r\n* ...\r\n\r\n* Wahtever.",
          "timestamp": "2023-04-06T13:08:14+02:00",
          "tree_id": "d0be972572eeb665a13831d52cd8cfbc107218a4",
          "url": "https://github.com/huggingface/safetensors/commit/0b21039218bd869969838d60286fe6786a87c9c3"
        },
        "date": 1680779659669,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.5306601166054568,
            "unit": "iter/sec",
            "range": "stddev: 0.044319604621121104",
            "extra": "mean: 653.3129002000123 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.982273172210958,
            "unit": "iter/sec",
            "range": "stddev: 0.06440618860254309",
            "extra": "mean: 251.1128586000041 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.709869388810136,
            "unit": "iter/sec",
            "range": "stddev: 0.018211842551582928",
            "extra": "mean: 369.0214753999953 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.2448160301787827,
            "unit": "iter/sec",
            "range": "stddev: 0.054621768999031914",
            "extra": "mean: 445.47080320000987 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.28171668662008,
            "unit": "iter/sec",
            "range": "stddev: 0.010533706968611695",
            "extra": "mean: 233.55118360000233 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.40523164899778,
            "unit": "iter/sec",
            "range": "stddev: 0.0001731526060972663",
            "extra": "mean: 7.6684040000145615 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8207296804261244,
            "unit": "iter/sec",
            "range": "stddev: 0.034942819537611816",
            "extra": "mean: 549.2303502000141 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.6195743262377835,
            "unit": "iter/sec",
            "range": "stddev: 0.013703033969283839",
            "extra": "mean: 216.47016140000233 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a7061b4235b59312010b2dd6f9597381428ee9a2",
          "message": "Committing Cargo.lock. (#233)\n\n* Committing Cargo.lock.\r\n\r\n* Update .gitignore\r\n\r\nCo-authored-by: Mishig <dmishig@gmail.com>\r\n\r\n---------\r\n\r\nCo-authored-by: Mishig <dmishig@gmail.com>",
          "timestamp": "2023-04-12T16:18:58+02:00",
          "tree_id": "383eedf543da7026314bb7aaf34f457611c1f7ec",
          "url": "https://github.com/huggingface/safetensors/commit/a7061b4235b59312010b2dd6f9597381428ee9a2"
        },
        "date": 1681309462805,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4395514232006077,
            "unit": "iter/sec",
            "range": "stddev: 0.03137140511237757",
            "extra": "mean: 694.6608394000009 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.770497426017924,
            "unit": "iter/sec",
            "range": "stddev: 0.06943959103609092",
            "extra": "mean: 265.21699579997176 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.7259070780594366,
            "unit": "iter/sec",
            "range": "stddev: 0.012263079424173097",
            "extra": "mean: 366.85036259999606 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1101212327023426,
            "unit": "iter/sec",
            "range": "stddev: 0.010370246325913974",
            "extra": "mean: 473.90642039999875 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.343740459109852,
            "unit": "iter/sec",
            "range": "stddev: 0.010083318119079566",
            "extra": "mean: 230.21633299999849 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.6297093970895,
            "unit": "iter/sec",
            "range": "stddev: 0.00019707514030982976",
            "extra": "mean: 7.655226399992898 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.728867078213389,
            "unit": "iter/sec",
            "range": "stddev: 0.10746409685337421",
            "extra": "mean: 578.4134665999886 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.492568796639897,
            "unit": "iter/sec",
            "range": "stddev: 0.012840705799856245",
            "extra": "mean: 182.06417380001767 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ebffa60b4c5b2ff0ba841f621c5ca2083f9fb71e",
          "message": "Fixing  the logic to determine if PR is up-to-date. (#230)\n\n* Fixing  the logic to determine if PR is up-to-date.\r\n\r\n* Update bindings/python/convert.py",
          "timestamp": "2023-04-19T15:46:04+02:00",
          "tree_id": "f4b6c70b395ba10d25e0db7d2a288df1a1348a1f",
          "url": "https://github.com/huggingface/safetensors/commit/ebffa60b4c5b2ff0ba841f621c5ca2083f9fb71e"
        },
        "date": 1681912296407,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2476794942469702,
            "unit": "iter/sec",
            "range": "stddev: 0.006367970623698561",
            "extra": "mean: 801.4878857999861 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3854456219246543,
            "unit": "iter/sec",
            "range": "stddev: 0.06414948991389244",
            "extra": "mean: 295.38208899999745 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.0793970113975124,
            "unit": "iter/sec",
            "range": "stddev: 0.01818112335875595",
            "extra": "mean: 480.90864540000666 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8264810024862275,
            "unit": "iter/sec",
            "range": "stddev: 0.010248912243393674",
            "extra": "mean: 547.5009039999804 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.6656271080154648,
            "unit": "iter/sec",
            "range": "stddev: 0.010274105852827327",
            "extra": "mean: 272.8046171999722 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.60466319525736,
            "unit": "iter/sec",
            "range": "stddev: 0.0001891599761537972",
            "extra": "mean: 7.8367041999854345 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4253734501102546,
            "unit": "iter/sec",
            "range": "stddev: 0.026592024004743066",
            "extra": "mean: 701.5705252000089 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.325522534281662,
            "unit": "iter/sec",
            "range": "stddev: 0.011887511368644412",
            "extra": "mean: 231.1859416000175 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c36f8dba40f052c1166119b0a34f9a5a1f53f485",
          "message": "Fixing alignment (#235)\n\n* Alignment checking.\r\n\r\n* Fix alignments.\r\n\r\n* Updating python tests.\r\n\r\n* Improve a bit readability.",
          "timestamp": "2023-04-21T14:54:22+02:00",
          "tree_id": "64a317769fa34bbbd09aa07071a5e4ae82ceb70b",
          "url": "https://github.com/huggingface/safetensors/commit/c36f8dba40f052c1166119b0a34f9a5a1f53f485"
        },
        "date": 1682082039723,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0356651944295052,
            "unit": "iter/sec",
            "range": "stddev: 0.03100311377139359",
            "extra": "mean: 965.563007600008 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.06840642504395,
            "unit": "iter/sec",
            "range": "stddev: 0.07314151939912054",
            "extra": "mean: 325.90206820000276 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9126707780272818,
            "unit": "iter/sec",
            "range": "stddev: 0.03128192186787136",
            "extra": "mean: 522.8291305999846 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.7784961979180658,
            "unit": "iter/sec",
            "range": "stddev: 0.013253608323000227",
            "extra": "mean: 562.2727791999864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.4880596310191265,
            "unit": "iter/sec",
            "range": "stddev: 0.012711396796402135",
            "extra": "mean: 286.6923464000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 107.13511969169491,
            "unit": "iter/sec",
            "range": "stddev: 0.00019450505866375661",
            "extra": "mean: 9.334007399979782 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2489138130999533,
            "unit": "iter/sec",
            "range": "stddev: 0.02752421179697171",
            "extra": "mean: 800.6957641999975 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.8263327420368993,
            "unit": "iter/sec",
            "range": "stddev: 0.02160644938026851",
            "extra": "mean: 261.346847599998 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "11a0ddba9a80e4a50fd47957f1274c7e32baf667",
          "message": "Adding `save_model` and `load_model` to help with shared tensors with PyTorch. (#236)\n\n* Adding `save_model` and `load_model` to help with shared tensors\n\nand non contiguous tensors.\n\n* Tmp.\n\n* Adding more docs more tests on the edge cases.\n\n* Adding support for different `keep_name` on disk.\n\n* Torch 1.10 support + save_model failure test.\n\n* Fixes 1.10\n\n* Debugging on remote..\n\n* 1.\n\n* 2.\n\n* 3.\n\n* 4.\n\n* Trying to fix documentation builder version.\n\n* PR documentation + route from error message to new methods.\n\n* ..\n\n* Revert \"..\"\n\nThis reverts commit 564b6863fc11407e19982944733de999c17c95ba.\n\n* Fixing `.[dev]` ?\n\n* Adding contiguous support.\n\n* ?\n\n* Fix.",
          "timestamp": "2023-04-24T12:37:00+02:00",
          "tree_id": "6b86f5d98b7f6116880c8efdb9cdf281535e3cd4",
          "url": "https://github.com/huggingface/safetensors/commit/11a0ddba9a80e4a50fd47957f1274c7e32baf667"
        },
        "date": 1682333010446,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1689373147761006,
            "unit": "iter/sec",
            "range": "stddev: 0.02171279378979488",
            "extra": "mean: 855.4778663999969 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.2052219208859563,
            "unit": "iter/sec",
            "range": "stddev: 0.06689532589795358",
            "extra": "mean: 311.9908776000102 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9616377361463435,
            "unit": "iter/sec",
            "range": "stddev: 0.0323821810698414",
            "extra": "mean: 509.7781213999838 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.7929898472584398,
            "unit": "iter/sec",
            "range": "stddev: 0.016234050592795944",
            "extra": "mean: 557.7276422000068 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.3189663268689125,
            "unit": "iter/sec",
            "range": "stddev: 0.013315564100755605",
            "extra": "mean: 301.2986277999971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 66.5352331345146,
            "unit": "iter/sec",
            "range": "stddev: 0.0018097275342251137",
            "extra": "mean: 15.029630962264688 msec\nrounds: 53"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.404181426664744,
            "unit": "iter/sec",
            "range": "stddev: 0.07783796834406537",
            "extra": "mean: 712.1586861999958 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.4847558557082867,
            "unit": "iter/sec",
            "range": "stddev: 0.04887392689044787",
            "extra": "mean: 286.9641494000007 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a35d5e0353562e3a93cd6a53d073b531416f95eb",
          "message": "Fixing shared detection logic. (#232)\n\n* Fixing shared tensor logic.\r\n\r\n* Fixing shared detection logic.\r\n\r\n* torch 1.10 support.\r\n\r\n* Rebase.",
          "timestamp": "2023-04-24T16:15:32+02:00",
          "tree_id": "d85d039af99d3bee86779a1e8e22aba3b57c5f8d",
          "url": "https://github.com/huggingface/safetensors/commit/a35d5e0353562e3a93cd6a53d073b531416f95eb"
        },
        "date": 1682346068779,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.349817021274811,
            "unit": "iter/sec",
            "range": "stddev: 0.022029829063066227",
            "extra": "mean: 740.8411541999726 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9924031740923027,
            "unit": "iter/sec",
            "range": "stddev: 0.1022715628094214",
            "extra": "mean: 334.17956800000184 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.722989117343629,
            "unit": "iter/sec",
            "range": "stddev: 0.01867483293275715",
            "extra": "mean: 367.2434801999998 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1383469780683404,
            "unit": "iter/sec",
            "range": "stddev: 0.00962347359691775",
            "extra": "mean: 467.65095200000815 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.1816326287889884,
            "unit": "iter/sec",
            "range": "stddev: 0.010386645664188498",
            "extra": "mean: 239.14104579999957 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 132.3031271035464,
            "unit": "iter/sec",
            "range": "stddev: 0.00013012067440194443",
            "extra": "mean: 7.55840033332965 msec\nrounds: 9"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7850917872311014,
            "unit": "iter/sec",
            "range": "stddev: 0.06964292796518366",
            "extra": "mean: 560.1952836000237 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.567442587772225,
            "unit": "iter/sec",
            "range": "stddev: 0.017413191659973584",
            "extra": "mean: 218.94090199998573 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a7969d4683f014ef422cede384da1d1c3b1585bf",
          "message": "Fixing the spec with mention about the buffer which cannot be non-contiguous. (#253)\n\n* Adding specification that the byte buffer needs to be contiguous and\r\nfull.\r\n\r\n* Fix cargo.lock which messes up benchmarks.",
          "timestamp": "2023-05-22T16:07:24+02:00",
          "tree_id": "6d80c7580c9bac202293ad4ecd043b3a7f982f37",
          "url": "https://github.com/huggingface/safetensors/commit/a7969d4683f014ef422cede384da1d1c3b1585bf"
        },
        "date": 1684764826315,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2015814598041852,
            "unit": "iter/sec",
            "range": "stddev: 0.020336792666838664",
            "extra": "mean: 832.2365427999898 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.186298815172432,
            "unit": "iter/sec",
            "range": "stddev: 0.07881174031119137",
            "extra": "mean: 313.8437597999996 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.0870825928524175,
            "unit": "iter/sec",
            "range": "stddev: 0.015596097648608441",
            "extra": "mean: 479.137722400003 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8335707635727723,
            "unit": "iter/sec",
            "range": "stddev: 0.008982799052761562",
            "extra": "mean: 545.3839141999993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7084903373401397,
            "unit": "iter/sec",
            "range": "stddev: 0.009869543441089437",
            "extra": "mean: 269.6515047999924 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 126.23782818000565,
            "unit": "iter/sec",
            "range": "stddev: 0.00019493662056173434",
            "extra": "mean: 7.921555800010083 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.505642230595871,
            "unit": "iter/sec",
            "range": "stddev: 0.09600018494426561",
            "extra": "mean: 664.1684057999896 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.2196899919946222,
            "unit": "iter/sec",
            "range": "stddev: 0.008564114753552489",
            "extra": "mean: 310.5889083999955 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e25c212d7630f8e973f9b12e17ccdfdfb8476718",
          "message": "Update metadata_parsing.mdx (#261)\n\nAdd links to featured models",
          "timestamp": "2023-06-02T10:26:28+02:00",
          "tree_id": "2594190c2cdb4fed121780c77b937899fd8940d0",
          "url": "https://github.com/huggingface/safetensors/commit/e25c212d7630f8e973f9b12e17ccdfdfb8476718"
        },
        "date": 1685694782967,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0765534438880846,
            "unit": "iter/sec",
            "range": "stddev: 0.054273827402864204",
            "extra": "mean: 928.8902522000171 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.5630711106815522,
            "unit": "iter/sec",
            "range": "stddev: 0.08538937302168316",
            "extra": "mean: 390.1569471999892 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7440239402975648,
            "unit": "iter/sec",
            "range": "stddev: 0.020926588225467044",
            "extra": "mean: 573.3866243999955 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5501630876204266,
            "unit": "iter/sec",
            "range": "stddev: 0.00768894061000977",
            "extra": "mean: 645.0934149999966 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.796297222409278,
            "unit": "iter/sec",
            "range": "stddev: 0.015422303840943586",
            "extra": "mean: 357.61577559999296 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 73.06907978576217,
            "unit": "iter/sec",
            "range": "stddev: 0.0006789139228373459",
            "extra": "mean: 13.685679399986839 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2049901290293386,
            "unit": "iter/sec",
            "range": "stddev: 0.1205119357588575",
            "extra": "mean: 829.8823168000013 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.81463826644197,
            "unit": "iter/sec",
            "range": "stddev: 0.04238965890905638",
            "extra": "mean: 355.28544179999244 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "alvarobartt@gmail.com",
            "name": "Alvaro Bartolome",
            "username": "alvarobartt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c18150a866f37182e83dccdf40e779cabe6cb3da",
          "message": "Add `safejax` in \"Featured Projects\" (#260)",
          "timestamp": "2023-06-05T16:40:06+02:00",
          "tree_id": "785462ed3a3c815a422df90469a6f111d959fcd5",
          "url": "https://github.com/huggingface/safetensors/commit/c18150a866f37182e83dccdf40e779cabe6cb3da"
        },
        "date": 1685976306214,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3586988938522817,
            "unit": "iter/sec",
            "range": "stddev: 0.048074126052301686",
            "extra": "mean: 735.998243999984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.5912345067714146,
            "unit": "iter/sec",
            "range": "stddev: 0.0814076688300771",
            "extra": "mean: 278.45577840000715 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6389701875701244,
            "unit": "iter/sec",
            "range": "stddev: 0.018238541068390908",
            "extra": "mean: 378.93569420000404 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1178819479776423,
            "unit": "iter/sec",
            "range": "stddev: 0.011081289610172957",
            "extra": "mean: 472.1698491999973 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.340648292803091,
            "unit": "iter/sec",
            "range": "stddev: 0.011535064029065022",
            "extra": "mean: 230.38033320000295 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 127.94317276036267,
            "unit": "iter/sec",
            "range": "stddev: 0.00017376723008237494",
            "extra": "mean: 7.815970000001471 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6819112517992798,
            "unit": "iter/sec",
            "range": "stddev: 0.04865552244427056",
            "extra": "mean: 594.5616921999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.469234669104372,
            "unit": "iter/sec",
            "range": "stddev: 0.0070147199955995256",
            "extra": "mean: 288.24801299998626 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a19455e0cf6d2ceb44e56fc9605cab556661530",
          "message": "Adding error checking, convertion can fail on schedulers. (#266)\n\n* Adding error checking, convertion can fail on schedulers.\r\n\r\n* Printing messages on using the script directly.",
          "timestamp": "2023-06-06T13:03:30+02:00",
          "tree_id": "3ada30a7ceef1bf9f0b4f1e1ee46b0139f2bac83",
          "url": "https://github.com/huggingface/safetensors/commit/0a19455e0cf6d2ceb44e56fc9605cab556661530"
        },
        "date": 1686049787388,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.064229498927225,
            "unit": "iter/sec",
            "range": "stddev: 0.04812963191486314",
            "extra": "mean: 939.6469474000014 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.4794138323343415,
            "unit": "iter/sec",
            "range": "stddev: 0.08446827281457961",
            "extra": "mean: 403.32113459999164 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.643871064423616,
            "unit": "iter/sec",
            "range": "stddev: 0.02450543851825803",
            "extra": "mean: 608.3202153999991 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5278812634675565,
            "unit": "iter/sec",
            "range": "stddev: 0.010086295662051641",
            "extra": "mean: 654.5011212000077 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.75768080948692,
            "unit": "iter/sec",
            "range": "stddev: 0.08024852738232222",
            "extra": "mean: 266.1215921999883 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 69.8855074118822,
            "unit": "iter/sec",
            "range": "stddev: 0.0013319738629917744",
            "extra": "mean: 14.309118399989984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.1045502353863117,
            "unit": "iter/sec",
            "range": "stddev: 0.05850692908175718",
            "extra": "mean: 905.3458755999941 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.780447821739983,
            "unit": "iter/sec",
            "range": "stddev: 0.07267435297298987",
            "extra": "mean: 359.65429459999996 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "davanstrien@users.noreply.github.com",
            "name": "Daniel van Strien",
            "username": "davanstrien"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fb637cf166520ded5096bee621c421d8ea9c434",
          "message": " Add BERTopic in Featured Projects (#267)",
          "timestamp": "2023-06-06T15:12:43+02:00",
          "tree_id": "112e5ec5e63bd62ac18e16611ef72692017ef1b8",
          "url": "https://github.com/huggingface/safetensors/commit/5fb637cf166520ded5096bee621c421d8ea9c434"
        },
        "date": 1686057459466,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3552360343505172,
            "unit": "iter/sec",
            "range": "stddev: 0.021013918787218377",
            "extra": "mean: 737.8788452000094 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.2693105897002486,
            "unit": "iter/sec",
            "range": "stddev: 0.095937896294104",
            "extra": "mean: 305.874884800005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.590621568065995,
            "unit": "iter/sec",
            "range": "stddev: 0.019506157001830813",
            "extra": "mean: 386.00774899999806 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1370153165810786,
            "unit": "iter/sec",
            "range": "stddev: 0.009743745718796371",
            "extra": "mean: 467.9423644000167 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.620269144021106,
            "unit": "iter/sec",
            "range": "stddev: 0.011324343612423628",
            "extra": "mean: 216.43760759999395 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.15647166434985,
            "unit": "iter/sec",
            "range": "stddev: 0.00015523181930568377",
            "extra": "mean: 7.624480799995581 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0857603405778886,
            "unit": "iter/sec",
            "range": "stddev: 0.049389430402830246",
            "extra": "mean: 479.4414682000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.152876875429135,
            "unit": "iter/sec",
            "range": "stddev: 0.017043591210342754",
            "extra": "mean: 194.0663486000176 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ishizaki@jp.ibm.com",
            "name": "Kazuaki Ishizaki",
            "username": "kiszk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "32b3c4dcba12b9f02880607056274c4cb2a783ca",
          "message": "fix typos (#257)",
          "timestamp": "2023-06-06T15:14:50+02:00",
          "tree_id": "3a009074db81180c4a3bdfff78031c8be29cb3d4",
          "url": "https://github.com/huggingface/safetensors/commit/32b3c4dcba12b9f02880607056274c4cb2a783ca"
        },
        "date": 1686057680556,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.109908355222258,
            "unit": "iter/sec",
            "range": "stddev: 0.03390356363256735",
            "extra": "mean: 900.9752880000178 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.9107698987488178,
            "unit": "iter/sec",
            "range": "stddev: 0.07443035507576984",
            "extra": "mean: 343.55171820000123 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9265346311251055,
            "unit": "iter/sec",
            "range": "stddev: 0.02527600935312432",
            "extra": "mean: 519.0667137999981 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.6840465621535357,
            "unit": "iter/sec",
            "range": "stddev: 0.020837200448099412",
            "extra": "mean: 593.8078094000048 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.201534614215258,
            "unit": "iter/sec",
            "range": "stddev: 0.037662466463584475",
            "extra": "mean: 238.0082735999963 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 61.895830159657734,
            "unit": "iter/sec",
            "range": "stddev: 0.0003242703565599348",
            "extra": "mean: 16.15617719999136 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.15837549640013,
            "unit": "iter/sec",
            "range": "stddev: 0.04116162737208289",
            "extra": "mean: 863.2779293999988 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.0794533737455687,
            "unit": "iter/sec",
            "range": "stddev: 0.0542213210927045",
            "extra": "mean: 324.732956999992 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "49c45120194c4de27cdf2b62bbb84f3a8a19592e",
          "message": "Adding issue templates. (#268)",
          "timestamp": "2023-06-07T13:45:47+02:00",
          "tree_id": "1048ef0dcfc3324815ccc9312797db314ffc7663",
          "url": "https://github.com/huggingface/safetensors/commit/49c45120194c4de27cdf2b62bbb84f3a8a19592e"
        },
        "date": 1686138707805,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0161081065670399,
            "unit": "iter/sec",
            "range": "stddev: 0.03956117631028778",
            "extra": "mean: 984.1472512000109 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.8360534885653625,
            "unit": "iter/sec",
            "range": "stddev: 0.08555161697989944",
            "extra": "mean: 352.6026586000171 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.8919561108138414,
            "unit": "iter/sec",
            "range": "stddev: 0.034012336659935395",
            "extra": "mean: 528.5534871999971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.7345378708078174,
            "unit": "iter/sec",
            "range": "stddev: 0.012609531611354528",
            "extra": "mean: 576.5224367999963 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.4639744341453285,
            "unit": "iter/sec",
            "range": "stddev: 0.014254593903744311",
            "extra": "mean: 288.68573339997283 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 101.39685318939254,
            "unit": "iter/sec",
            "range": "stddev: 0.0008437463536838407",
            "extra": "mean: 9.86223899998322 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.4664855229335059,
            "unit": "iter/sec",
            "range": "stddev: 0.07247409633698972",
            "extra": "mean: 681.9024016000071 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.680881439266093,
            "unit": "iter/sec",
            "range": "stddev: 0.01276776829173976",
            "extra": "mean: 373.0116465999913 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c18bd7cf450dcb27a76aaa2e35b182718d50b91f",
          "message": "Remove ISSUE links. (#269)",
          "timestamp": "2023-06-07T13:56:51+02:00",
          "tree_id": "a0b76dc18a446482c195beb1727b8d2579d44844",
          "url": "https://github.com/huggingface/safetensors/commit/c18bd7cf450dcb27a76aaa2e35b182718d50b91f"
        },
        "date": 1686139307477,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2983788626669275,
            "unit": "iter/sec",
            "range": "stddev: 0.03171025715707708",
            "extra": "mean: 770.1912198000173 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6551260515385464,
            "unit": "iter/sec",
            "range": "stddev: 0.09088490874974556",
            "extra": "mean: 376.62995299998556 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.096022068416085,
            "unit": "iter/sec",
            "range": "stddev: 0.017147351911824094",
            "extra": "mean: 477.0942133999938 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.8554453760631022,
            "unit": "iter/sec",
            "range": "stddev: 0.009995486641285578",
            "extra": "mean: 538.9541578000035 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.6954925273506123,
            "unit": "iter/sec",
            "range": "stddev: 0.010900536382004656",
            "extra": "mean: 270.5999247999898 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 129.00826086301865,
            "unit": "iter/sec",
            "range": "stddev: 0.00021596726635707197",
            "extra": "mean: 7.7514416000212805 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6624077868261653,
            "unit": "iter/sec",
            "range": "stddev: 0.06308129361653017",
            "extra": "mean: 601.5371245999631 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.178464536786524,
            "unit": "iter/sec",
            "range": "stddev: 0.03671427378526336",
            "extra": "mean: 314.6173217999831 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d593c38a1c7c007bc195bea11fabe383b0fa04e1",
          "message": "[doc build] Use secrets (#270)",
          "timestamp": "2023-06-09T12:59:02+02:00",
          "tree_id": "9db1e1fb54d373847b2122af6e04a238d01152c7",
          "url": "https://github.com/huggingface/safetensors/commit/d593c38a1c7c007bc195bea11fabe383b0fa04e1"
        },
        "date": 1686308708105,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0716444210527376,
            "unit": "iter/sec",
            "range": "stddev: 0.06626673397282028",
            "extra": "mean: 933.1453422000209 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6733175270832463,
            "unit": "iter/sec",
            "range": "stddev: 0.07546908971485283",
            "extra": "mean: 374.0670496000007 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7338713653157525,
            "unit": "iter/sec",
            "range": "stddev: 0.0352399602657175",
            "extra": "mean: 576.7440538000301 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5689441109086995,
            "unit": "iter/sec",
            "range": "stddev: 0.01792731698374936",
            "extra": "mean: 637.3713334000286 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.00705826398439,
            "unit": "iter/sec",
            "range": "stddev: 0.019065096215613364",
            "extra": "mean: 332.5509226000122 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 64.15968495824163,
            "unit": "iter/sec",
            "range": "stddev: 0.002321011087704954",
            "extra": "mean: 15.586111444450681 msec\nrounds: 9"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.234688137909176,
            "unit": "iter/sec",
            "range": "stddev: 0.09115963714010993",
            "extra": "mean: 809.9211204000085 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.937280052834674,
            "unit": "iter/sec",
            "range": "stddev: 0.0735233671257553",
            "extra": "mean: 340.45102340000994 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "24695242+thomasw21@users.noreply.github.com",
            "name": "Thomas Wang",
            "username": "thomasw21"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3dab13f0d7d461c9342bb82203b8c826ccb87f7c",
          "message": "Make tensor identifier a bit more robust, and transformers compatible (#264)\n\n* Make tensor identifier a bit more robust, and transformers compatible\r\n\r\n* Prove that you need to storage storage size as well\r\n\r\n* Lint\r\n\r\n* Backward compatility for torch version\r\n\r\n* Lint\r\n\r\n* Fixing tests.\r\n\r\n* Test supporting mutliple torch versions.\r\n\r\n* Fixing _flatten share pointer detection.\r\n\r\n* More code reuse.\r\n\r\n* Better error message.\r\n\r\n* Attempting to debug what's wrong.\r\n\r\n* Fixing torch 1.13... God this is awful.\r\n\r\n* Latest python ?\r\n\r\n* Dancing with versions.\r\n\r\n* ..\r\n\r\n---------\r\n\r\nCo-authored-by: Nicolas Patry <patry.nicolas@protonmail.com>",
          "timestamp": "2023-06-13T18:20:19+02:00",
          "tree_id": "7defeaa79968ebc34384d5d30fd93518c9b6f511",
          "url": "https://github.com/huggingface/safetensors/commit/3dab13f0d7d461c9342bb82203b8c826ccb87f7c"
        },
        "date": 1686673517137,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3799934488178422,
            "unit": "iter/sec",
            "range": "stddev: 0.009860233133797698",
            "extra": "mean: 724.6411211999884 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.281282915574096,
            "unit": "iter/sec",
            "range": "stddev: 0.09826760850444494",
            "extra": "mean: 304.75884760002145 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6385483158463674,
            "unit": "iter/sec",
            "range": "stddev: 0.01753233353892717",
            "extra": "mean: 378.99628139999777 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.123502558870727,
            "unit": "iter/sec",
            "range": "stddev: 0.012763573201802346",
            "extra": "mean: 470.9200823999936 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.189494432809798,
            "unit": "iter/sec",
            "range": "stddev: 0.011733485055256593",
            "extra": "mean: 238.69228519998842 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 123.52632476908455,
            "unit": "iter/sec",
            "range": "stddev: 0.0006157686745168388",
            "extra": "mean: 8.095440400006737 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.0335806184311447,
            "unit": "iter/sec",
            "range": "stddev: 0.05336112861551083",
            "extra": "mean: 491.74347500000977 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.615243075109603,
            "unit": "iter/sec",
            "range": "stddev: 0.04486243158450908",
            "extra": "mean: 216.67331140001806 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "laurent.mazare@gmail.com",
            "name": "Laurent Mazare",
            "username": "LaurentMazare"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eb19add1938041846f6a9c2fea99c254f9c529fe",
          "message": "Add a reference to tch-rs (#274)",
          "timestamp": "2023-06-15T17:00:03+02:00",
          "tree_id": "dc4d05b0491e938f37d7703eab35c5e1495fe25b",
          "url": "https://github.com/huggingface/safetensors/commit/eb19add1938041846f6a9c2fea99c254f9c529fe"
        },
        "date": 1686841538365,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.086699447156886,
            "unit": "iter/sec",
            "range": "stddev: 0.029078442277605757",
            "extra": "mean: 920.2176394000048 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6844509076763536,
            "unit": "iter/sec",
            "range": "stddev: 0.09860336909681866",
            "extra": "mean: 372.51565940000546 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.874939248218515,
            "unit": "iter/sec",
            "range": "stddev: 0.021565455129571687",
            "extra": "mean: 533.3506143999898 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.6155283337545534,
            "unit": "iter/sec",
            "range": "stddev: 0.012104440941996314",
            "extra": "mean: 618.9925481999808 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.1616691534522223,
            "unit": "iter/sec",
            "range": "stddev: 0.015028716910860758",
            "extra": "mean: 316.2886284000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 112.65135300701256,
            "unit": "iter/sec",
            "range": "stddev: 0.0004222004633951983",
            "extra": "mean: 8.876946199995928 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3910908338180912,
            "unit": "iter/sec",
            "range": "stddev: 0.11121192423321578",
            "extra": "mean: 718.8603185999909 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.292790267053628,
            "unit": "iter/sec",
            "range": "stddev: 0.05724089127388587",
            "extra": "mean: 303.6938033999945 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jakub.kaczmarzyk@gmail.com",
            "name": "Jakub Kaczmarzyk",
            "username": "kaczmarj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "736ba7d548c4c0c8c545cdcb29e51c98d69d829b",
          "message": "fix syntax error in import statement (#276)",
          "timestamp": "2023-06-21T12:21:48+02:00",
          "tree_id": "85ce5b3057c827c61cd808bcfa3d87f339e31971",
          "url": "https://github.com/huggingface/safetensors/commit/736ba7d548c4c0c8c545cdcb29e51c98d69d829b"
        },
        "date": 1687343223527,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3687533454863727,
            "unit": "iter/sec",
            "range": "stddev: 0.011064213770666945",
            "extra": "mean: 730.5918216000123 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.4951589308601796,
            "unit": "iter/sec",
            "range": "stddev: 0.08539536536344235",
            "extra": "mean: 286.1100224000097 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.5835289716675254,
            "unit": "iter/sec",
            "range": "stddev: 0.024793337542674793",
            "extra": "mean: 387.0674612000016 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.034064936235996,
            "unit": "iter/sec",
            "range": "stddev: 0.0071328172346106986",
            "extra": "mean: 491.6263891999847 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.06677325120843,
            "unit": "iter/sec",
            "range": "stddev: 0.013493808441402987",
            "extra": "mean: 245.89519460000702 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 129.41655342281774,
            "unit": "iter/sec",
            "range": "stddev: 0.00016051800599900266",
            "extra": "mean: 7.726986800003033 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.7765736436398523,
            "unit": "iter/sec",
            "range": "stddev: 0.10216522382757436",
            "extra": "mean: 562.8812537999806 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.838633451772902,
            "unit": "iter/sec",
            "range": "stddev: 0.04175281041286389",
            "extra": "mean: 206.66992240001036 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a29fac04503addcd7cea4e57ef949646efeb83d5",
          "message": "Missing files. (#282)\n\n* Update README.md\r\n\r\n* Missing files.\r\n\r\n---------\r\n\r\nCo-authored-by: iacore <74560659+iacore@users.noreply.github.com>",
          "timestamp": "2023-06-27T11:39:55+02:00",
          "tree_id": "ab653926c621d4c89b3f80659e56799251fab7d3",
          "url": "https://github.com/huggingface/safetensors/commit/a29fac04503addcd7cea4e57ef949646efeb83d5"
        },
        "date": 1687859120391,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.427023278334092,
            "unit": "iter/sec",
            "range": "stddev: 0.028060272561449526",
            "extra": "mean: 700.7594165999876 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.8835150593912817,
            "unit": "iter/sec",
            "range": "stddev: 0.06176652092238144",
            "extra": "mean: 257.49867960000756 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6314428920761777,
            "unit": "iter/sec",
            "range": "stddev: 0.02055074541408378",
            "extra": "mean: 380.0196473999904 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.5969451586662227,
            "unit": "iter/sec",
            "range": "stddev: 0.009585687339772032",
            "extra": "mean: 385.06781580000506 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 6.7683900460526525,
            "unit": "iter/sec",
            "range": "stddev: 0.03425166607428438",
            "extra": "mean: 147.74562239999796 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 129.1900733242505,
            "unit": "iter/sec",
            "range": "stddev: 0.00016826485094422567",
            "extra": "mean: 7.740532799994071 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8896026663502725,
            "unit": "iter/sec",
            "range": "stddev: 0.08553604086703896",
            "extra": "mean: 529.2117850000068 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.254059605634832,
            "unit": "iter/sec",
            "range": "stddev: 0.014573016902270008",
            "extra": "mean: 190.32901700002185 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e9671260d5b0f3da2f4355cd88061bdb6c048f07",
          "message": "Fixing uniform empty tensor handling (#283)\n\n- Fixes #280",
          "timestamp": "2023-06-30T09:53:38+02:00",
          "tree_id": "9ae03cf98c627cac66be2d52e4ece810abc72cfb",
          "url": "https://github.com/huggingface/safetensors/commit/e9671260d5b0f3da2f4355cd88061bdb6c048f07"
        },
        "date": 1688111963204,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0016832260890696,
            "unit": "iter/sec",
            "range": "stddev: 0.04380945965913922",
            "extra": "mean: 998.3196024000108 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.520296447571047,
            "unit": "iter/sec",
            "range": "stddev: 0.08698577473890125",
            "extra": "mean: 396.7787206000139 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.7078629523242217,
            "unit": "iter/sec",
            "range": "stddev: 0.025530149653298407",
            "extra": "mean: 585.5270756000095 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5603763765132783,
            "unit": "iter/sec",
            "range": "stddev: 0.01203861990672288",
            "extra": "mean: 640.87102000002 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.689900086889094,
            "unit": "iter/sec",
            "range": "stddev: 0.014572585983765034",
            "extra": "mean: 371.7610200000081 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 73.81426154276276,
            "unit": "iter/sec",
            "range": "stddev: 0.0005507887651441767",
            "extra": "mean: 13.547517499997893 msec\nrounds: 18"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.2429390787435912,
            "unit": "iter/sec",
            "range": "stddev: 0.06492014538069724",
            "extra": "mean: 804.5446611999978 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.346884189113292,
            "unit": "iter/sec",
            "range": "stddev: 0.03841157202181318",
            "extra": "mean: 298.78536079999094 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eca6cf10c3a782147c73569edd18649b62e3a840",
          "message": "Superseed #279 (#285)\n\n* Update README.md\r\n\r\nAdd note about arbitrary JSON not allowed in __metadata__.\r\n\r\n* Update README.md\r\n\r\nCo-authored-by: Julien Chaumond <julien@huggingface.co>\r\n\r\n* Update other readmes.\r\n\r\n---------\r\n\r\nCo-authored-by: by321 <by321@hotmail.com>\r\nCo-authored-by: Julien Chaumond <julien@huggingface.co>",
          "timestamp": "2023-06-30T10:32:00+02:00",
          "tree_id": "dbd11a8be10ad98805fb8d22916d221306117275",
          "url": "https://github.com/huggingface/safetensors/commit/eca6cf10c3a782147c73569edd18649b62e3a840"
        },
        "date": 1688114210882,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4181213056358692,
            "unit": "iter/sec",
            "range": "stddev: 0.019943472363111548",
            "extra": "mean: 705.1582936000045 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.974254054450939,
            "unit": "iter/sec",
            "range": "stddev: 0.10356178828301939",
            "extra": "mean: 336.21875659999887 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.673156037371277,
            "unit": "iter/sec",
            "range": "stddev: 0.0169801057916223",
            "extra": "mean: 374.08964759998753 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.0862521548283337,
            "unit": "iter/sec",
            "range": "stddev: 0.008194799141992302",
            "extra": "mean: 479.3284443999937 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.346133785603806,
            "unit": "iter/sec",
            "range": "stddev: 0.010216466645098624",
            "extra": "mean: 230.08955760000163 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 130.89906451643276,
            "unit": "iter/sec",
            "range": "stddev: 0.00015407100716701815",
            "extra": "mean: 7.639474000018254 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 2.052160225998428,
            "unit": "iter/sec",
            "range": "stddev: 0.06870353657830679",
            "extra": "mean: 487.2913856000082 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.9131120301988433,
            "unit": "iter/sec",
            "range": "stddev: 0.008921439517164464",
            "extra": "mean: 255.5510786000127 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "025d6ac6506a763ec239ca08efe64e75eabb6ab5",
          "message": "Fix failing doc build workflows (#287)\n\n* [wip: fix docs]\r\n\r\n* wip\r\n\r\n* pin tf version for doc-builder",
          "timestamp": "2023-06-30T12:30:41+02:00",
          "tree_id": "8d61336828f294939fda6e6b50d98c462065f238",
          "url": "https://github.com/huggingface/safetensors/commit/025d6ac6506a763ec239ca08efe64e75eabb6ab5"
        },
        "date": 1688121339823,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.351845936244109,
            "unit": "iter/sec",
            "range": "stddev: 0.0323955130091406",
            "extra": "mean: 739.7292644000117 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.87885671688186,
            "unit": "iter/sec",
            "range": "stddev: 0.05770479423086241",
            "extra": "mean: 257.80792459997883 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.5820147378684086,
            "unit": "iter/sec",
            "range": "stddev: 0.0238192525471205",
            "extra": "mean: 387.29445860001306 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.0153869042254096,
            "unit": "iter/sec",
            "range": "stddev: 0.009576193290964324",
            "extra": "mean: 496.1826426000016 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.723259332940233,
            "unit": "iter/sec",
            "range": "stddev: 0.011056435254579925",
            "extra": "mean: 211.71820759998354 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 132.51900229695764,
            "unit": "iter/sec",
            "range": "stddev: 0.00017534486153600976",
            "extra": "mean: 7.546087600019291 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.647710114953311,
            "unit": "iter/sec",
            "range": "stddev: 0.03513150094536201",
            "extra": "mean: 606.9028714000069 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7763325110440507,
            "unit": "iter/sec",
            "range": "stddev: 0.014724678303772972",
            "extra": "mean: 264.80718980001257 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmishig@gmail.com",
            "name": "Mishig",
            "username": "mishig25"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "33a6557c1f85c2e5543032af5049fb19916d557c",
          "message": "Revert dev change (#288)",
          "timestamp": "2023-06-30T15:13:17+02:00",
          "tree_id": "59c15f9cec1ef6981d0f080ad4261e67a4e718a2",
          "url": "https://github.com/huggingface/safetensors/commit/33a6557c1f85c2e5543032af5049fb19916d557c"
        },
        "date": 1688131143836,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4175386756475299,
            "unit": "iter/sec",
            "range": "stddev: 0.03558353414389897",
            "extra": "mean: 705.448124399993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3433117257883103,
            "unit": "iter/sec",
            "range": "stddev: 0.0906313356094979",
            "extra": "mean: 299.1046249999954 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.631144870781406,
            "unit": "iter/sec",
            "range": "stddev: 0.01665068025929769",
            "extra": "mean: 380.0626909999892 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.025365292534698,
            "unit": "iter/sec",
            "range": "stddev: 0.010465410525370637",
            "extra": "mean: 493.7380944000097 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.048670076779747,
            "unit": "iter/sec",
            "range": "stddev: 0.011251156037640151",
            "extra": "mean: 246.99468740001294 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 131.97008586522173,
            "unit": "iter/sec",
            "range": "stddev: 0.00018485310161612048",
            "extra": "mean: 7.5774748000185355 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.683643496736696,
            "unit": "iter/sec",
            "range": "stddev: 0.11775243898413276",
            "extra": "mean: 593.9499674000103 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.200395384813524,
            "unit": "iter/sec",
            "range": "stddev: 0.0308813858538549",
            "extra": "mean: 192.2930711999811 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "amitani.tky@gmail.com",
            "name": "Akinori Mitani",
            "username": "amitani"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b6b4b5bf81f36a3072c7e473e2c24346fdc4ef07",
          "message": "Fix docstring Args names. (#290)\n\nFix docstring Args names.",
          "timestamp": "2023-07-04T16:41:59+02:00",
          "tree_id": "0bea566cd622169fa242360aa65179e6fa024bcd",
          "url": "https://github.com/huggingface/safetensors/commit/b6b4b5bf81f36a3072c7e473e2c24346fdc4ef07"
        },
        "date": 1688482044238,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1187332998835295,
            "unit": "iter/sec",
            "range": "stddev: 0.041170924498897525",
            "extra": "mean: 893.8680917999932 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.089168663169708,
            "unit": "iter/sec",
            "range": "stddev: 0.07244453769094493",
            "extra": "mean: 323.7116871999888 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.9211167817632613,
            "unit": "iter/sec",
            "range": "stddev: 0.019859365174132514",
            "extra": "mean: 520.5305630000112 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.699561157333311,
            "unit": "iter/sec",
            "range": "stddev: 0.008056388241393962",
            "extra": "mean: 588.3871819999968 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.90386496469496,
            "unit": "iter/sec",
            "range": "stddev: 0.05970069358965284",
            "extra": "mean: 203.9207863999991 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 125.16982677000081,
            "unit": "iter/sec",
            "range": "stddev: 0.00014460240782572216",
            "extra": "mean: 7.989145833344461 msec\nrounds: 12"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5563082795488719,
            "unit": "iter/sec",
            "range": "stddev: 0.05744878961770627",
            "extra": "mean: 642.5462186000004 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.030923061807142,
            "unit": "iter/sec",
            "range": "stddev: 0.010925323786412722",
            "extra": "mean: 248.08213519999072 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ab9f670b3adbd501b130a45f75460ece8f8f312a",
          "message": "Adding more control to the chosen weights saved on disk. (#292)",
          "timestamp": "2023-07-07T11:33:14+02:00",
          "tree_id": "c16d279c00f308bf72f19d8ca315ef17cfd9323f",
          "url": "https://github.com/huggingface/safetensors/commit/ab9f670b3adbd501b130a45f75460ece8f8f312a"
        },
        "date": 1688722679802,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.440021286325524,
            "unit": "iter/sec",
            "range": "stddev: 0.04057874644956364",
            "extra": "mean: 694.4341792000046 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.9293968087557873,
            "unit": "iter/sec",
            "range": "stddev: 0.06282106877629158",
            "extra": "mean: 254.49198660001005 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.639461203701558,
            "unit": "iter/sec",
            "range": "stddev: 0.017201502690453518",
            "extra": "mean: 378.8652012000057 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.101007631956846,
            "unit": "iter/sec",
            "range": "stddev: 0.012459989316869839",
            "extra": "mean: 475.96209780000436 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.279832833050623,
            "unit": "iter/sec",
            "range": "stddev: 0.011351997155236104",
            "extra": "mean: 233.65398580000374 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 133.75721172059463,
            "unit": "iter/sec",
            "range": "stddev: 0.00016896749178340362",
            "extra": "mean: 7.476232400006211 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8023837821072002,
            "unit": "iter/sec",
            "range": "stddev: 0.089823698836269",
            "extra": "mean: 554.8207933999947 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 5.459055551552839,
            "unit": "iter/sec",
            "range": "stddev: 0.021506448768039175",
            "extra": "mean: 183.18186919998425 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "35296208+wouterzwerink@users.noreply.github.com",
            "name": "Wouter Zwerink",
            "username": "wouterzwerink"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c3c2b5d9b2cb33395370527d2039097b5344ace0",
          "message": "Autodoc for save_model and load_model (#293)",
          "timestamp": "2023-07-07T13:21:02+02:00",
          "tree_id": "ca15d616291834072392a20cedea4cdae1141489",
          "url": "https://github.com/huggingface/safetensors/commit/c3c2b5d9b2cb33395370527d2039097b5344ace0"
        },
        "date": 1688729152509,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.3835764376982496,
            "unit": "iter/sec",
            "range": "stddev: 0.038311118347537865",
            "extra": "mean: 722.7645489999986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.8718367639566447,
            "unit": "iter/sec",
            "range": "stddev: 0.06703628067194292",
            "extra": "mean: 258.27535120000675 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.6175763987651637,
            "unit": "iter/sec",
            "range": "stddev: 0.010956724280079044",
            "extra": "mean: 382.0327844000076 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.099649731518625,
            "unit": "iter/sec",
            "range": "stddev: 0.010376097450830197",
            "extra": "mean: 476.2699153999961 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.096376941553875,
            "unit": "iter/sec",
            "range": "stddev: 0.010469968608159911",
            "extra": "mean: 244.11815959999785 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 132.98319036642295,
            "unit": "iter/sec",
            "range": "stddev: 0.00018865724393112994",
            "extra": "mean: 7.519747399987864 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8818210473355348,
            "unit": "iter/sec",
            "range": "stddev: 0.039369272669760834",
            "extra": "mean: 531.4001569999959 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.7441106898583474,
            "unit": "iter/sec",
            "range": "stddev: 0.011345290993111245",
            "extra": "mean: 267.0861207999792 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "35296208+wouterzwerink@users.noreply.github.com",
            "name": "Wouter Zwerink",
            "username": "wouterzwerink"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0c354d9e4d46f208c4736c259756f9c866971fb0",
          "message": "Fix link to shared tensor page (#294)",
          "timestamp": "2023-07-07T15:05:57+02:00",
          "tree_id": "5353328e140af1823a3bb696acb20ba2269656aa",
          "url": "https://github.com/huggingface/safetensors/commit/0c354d9e4d46f208c4736c259756f9c866971fb0"
        },
        "date": 1688735523758,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.2264018569282105,
            "unit": "iter/sec",
            "range": "stddev: 0.04533684677718612",
            "extra": "mean: 815.393416400002 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.355036872267793,
            "unit": "iter/sec",
            "range": "stddev: 0.06385106804842247",
            "extra": "mean: 298.05931740000915 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.0367577840418476,
            "unit": "iter/sec",
            "range": "stddev: 0.017348507700494762",
            "extra": "mean: 490.97639780001145 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.0154971511456,
            "unit": "iter/sec",
            "range": "stddev: 0.010246544545512338",
            "extra": "mean: 496.155501599992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.7211944477484225,
            "unit": "iter/sec",
            "range": "stddev: 0.009878457806680264",
            "extra": "mean: 268.730918000017 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 124.54968128607376,
            "unit": "iter/sec",
            "range": "stddev: 0.00014901280167649284",
            "extra": "mean: 8.028924600000664 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.371921831231722,
            "unit": "iter/sec",
            "range": "stddev: 0.09085663848789541",
            "extra": "mean: 728.9045025999712 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.86551849109108,
            "unit": "iter/sec",
            "range": "stddev: 0.050631375151043725",
            "extra": "mean: 258.6975077999796 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d93a71ccbf200ef15574d7d5e29d546d85c88a9",
          "message": "Support musicgen conversion. (#296)",
          "timestamp": "2023-07-11T14:52:00+02:00",
          "tree_id": "6977b1fdee6da33bb6d9b19ab9db7b9458006666",
          "url": "https://github.com/huggingface/safetensors/commit/6d93a71ccbf200ef15574d7d5e29d546d85c88a9"
        },
        "date": 1689080291504,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0512294462349454,
            "unit": "iter/sec",
            "range": "stddev: 0.038132013270975984",
            "extra": "mean: 951.2671125999873 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.7005507141253537,
            "unit": "iter/sec",
            "range": "stddev: 0.09219978011087897",
            "extra": "mean: 370.29484199998706 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 1.834709294484418,
            "unit": "iter/sec",
            "range": "stddev: 0.02201076771377302",
            "extra": "mean: 545.0454755999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.5944739843161075,
            "unit": "iter/sec",
            "range": "stddev: 0.012332656164184956",
            "extra": "mean: 627.1660810000071 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.987213528821541,
            "unit": "iter/sec",
            "range": "stddev: 0.011678576270422777",
            "extra": "mean: 334.76013359999115 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 99.68714386152762,
            "unit": "iter/sec",
            "range": "stddev: 0.000481004600481227",
            "extra": "mean: 10.031383799992 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.313669199151495,
            "unit": "iter/sec",
            "range": "stddev: 0.0872185960480458",
            "extra": "mean: 761.2266471999988 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.606502381553515,
            "unit": "iter/sec",
            "range": "stddev: 0.011878143656643086",
            "extra": "mean: 277.27695539999786 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "39b28daf261058a9b2bca8464020c482d6bc99c2",
          "message": "Yaml is nice. (#325)",
          "timestamp": "2023-08-16T20:16:20+02:00",
          "tree_id": "5316eea856692b15e4c05c2c4eed832dbde720a7",
          "url": "https://github.com/huggingface/safetensors/commit/39b28daf261058a9b2bca8464020c482d6bc99c2"
        },
        "date": 1692210115790,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1351992869649818,
            "unit": "iter/sec",
            "range": "stddev: 0.04652590439749619",
            "extra": "mean: 880.9025969999993 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.0177843336046153,
            "unit": "iter/sec",
            "range": "stddev: 0.07621335830646529",
            "extra": "mean: 331.3689413999782 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.910757262795703,
            "unit": "iter/sec",
            "range": "stddev: 0.022085856939966755",
            "extra": "mean: 343.5532096000088 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.3351962476601726,
            "unit": "iter/sec",
            "range": "stddev: 0.025930889604221238",
            "extra": "mean: 428.2295336000061 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.536073093242702,
            "unit": "iter/sec",
            "range": "stddev: 0.03160975151262614",
            "extra": "mean: 180.6334531999937 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 109.6677759488284,
            "unit": "iter/sec",
            "range": "stddev: 0.0002109504464931743",
            "extra": "mean: 9.118448800006718 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6659353493691949,
            "unit": "iter/sec",
            "range": "stddev: 0.07204893791960498",
            "extra": "mean: 600.2633897999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.210365092606173,
            "unit": "iter/sec",
            "range": "stddev: 0.042419878831134045",
            "extra": "mean: 311.49105200000804 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "698dd6e137b9de9fc7a4ebea9adb6bb44ff6ddc1",
          "message": "Fixing big endian. (#327)\n\n* Fixing big endian.\r\n\r\n- Many more tests, with numpy coverage.\r\n- Actually test values on disk.\r\n\r\n* Re-enabling bf16 (we piggy back f16 byteswap).\r\n\r\n* Fmt.\r\n\r\n* Black.",
          "timestamp": "2023-08-17T08:44:48+02:00",
          "tree_id": "dc66afb29a612b54f64d27184be680f2a55a88b8",
          "url": "https://github.com/huggingface/safetensors/commit/698dd6e137b9de9fc7a4ebea9adb6bb44ff6ddc1"
        },
        "date": 1692255004971,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1653105174408873,
            "unit": "iter/sec",
            "range": "stddev: 0.04757159713279426",
            "extra": "mean: 858.140371200011 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.3328469731964336,
            "unit": "iter/sec",
            "range": "stddev: 0.05623381548396753",
            "extra": "mean: 300.0437787999999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 3.034004267272401,
            "unit": "iter/sec",
            "range": "stddev: 0.01990350583538217",
            "extra": "mean: 329.5974269999988 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.42383982591634,
            "unit": "iter/sec",
            "range": "stddev: 0.010499675092496982",
            "extra": "mean: 412.5685159999989 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.8289179367179615,
            "unit": "iter/sec",
            "range": "stddev: 0.022961884826894787",
            "extra": "mean: 261.1703923999926 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 120.28095128391823,
            "unit": "iter/sec",
            "range": "stddev: 0.0002279830517843209",
            "extra": "mean: 8.313868399989133 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5861923851230066,
            "unit": "iter/sec",
            "range": "stddev: 0.04654625072003018",
            "extra": "mean: 630.4405502000009 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.142204836215807,
            "unit": "iter/sec",
            "range": "stddev: 0.012328663871907317",
            "extra": "mean: 318.247871199992 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f4a6df0ffdfed0e73e9da5ae419b70f335f94ebe",
          "message": "Fixing boolean + numpy > 1.20 (#326)\n\n* Fixing boolean + numpy > 1.20\r\n\r\n* Adding bool numpy test.",
          "timestamp": "2023-08-17T08:58:26+02:00",
          "tree_id": "9bfb584cc110b410a387f10e17eb2a16e434f71d",
          "url": "https://github.com/huggingface/safetensors/commit/f4a6df0ffdfed0e73e9da5ae419b70f335f94ebe"
        },
        "date": 1692255885149,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 0.8885010394673902,
            "unit": "iter/sec",
            "range": "stddev: 0.052307815137623455",
            "extra": "mean: 1.1254910862000203 sec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.4776874434077643,
            "unit": "iter/sec",
            "range": "stddev: 0.08724830584584774",
            "extra": "mean: 403.60215840001956 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.3092866967823555,
            "unit": "iter/sec",
            "range": "stddev: 0.035377743101090514",
            "extra": "mean: 433.03414919998886 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 1.9666064235071756,
            "unit": "iter/sec",
            "range": "stddev: 0.021858704131849364",
            "extra": "mean: 508.4901524000088 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.753293169453254,
            "unit": "iter/sec",
            "range": "stddev: 0.05514182636161889",
            "extra": "mean: 266.43269119999786 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 66.31665795666929,
            "unit": "iter/sec",
            "range": "stddev: 0.0016481413572226559",
            "extra": "mean: 15.079167599992616 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.0686393498290592,
            "unit": "iter/sec",
            "range": "stddev: 0.04211852008060181",
            "extra": "mean: 935.76939699999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.423003855159103,
            "unit": "iter/sec",
            "range": "stddev: 0.012517490983671725",
            "extra": "mean: 412.7108579999913 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "clowman1993@gmail.com",
            "name": "Corey Lowman",
            "username": "coreylowman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "34a2a0a29014faa342ebfb1a0cc22b9574da96d6",
          "message": "impl `View` for `TensorView` (#329)\n\nThe existing impl only exists for `&TensorView`, which means you cannot pass a `Vec<(String, TensorView)>` into `serialize_to_file`. With this new impl, you can.",
          "timestamp": "2023-08-21T09:14:25+02:00",
          "tree_id": "e89e2d056a9f745f51efa9ad2e707198dc1901e7",
          "url": "https://github.com/huggingface/safetensors/commit/34a2a0a29014faa342ebfb1a0cc22b9574da96d6"
        },
        "date": 1692602430375,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1234651558888427,
            "unit": "iter/sec",
            "range": "stddev: 0.037431328136382565",
            "extra": "mean: 890.1032620000024 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.092953190943798,
            "unit": "iter/sec",
            "range": "stddev: 0.061855776614343556",
            "extra": "mean: 323.31559460001245 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.9480309547024866,
            "unit": "iter/sec",
            "range": "stddev: 0.023881604096171065",
            "extra": "mean: 339.20946399998684 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.4510971629062923,
            "unit": "iter/sec",
            "range": "stddev: 0.01910690619529209",
            "extra": "mean: 407.9805628000031 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.3122635249210175,
            "unit": "iter/sec",
            "range": "stddev: 0.03884329324300021",
            "extra": "mean: 188.24367340000663 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 100.78756613619653,
            "unit": "iter/sec",
            "range": "stddev: 0.00022200339080932007",
            "extra": "mean: 9.921858800009886 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3774366669372502,
            "unit": "iter/sec",
            "range": "stddev: 0.04344914499938044",
            "extra": "mean: 725.9861916000204 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.830856976743855,
            "unit": "iter/sec",
            "range": "stddev: 0.01175357509457722",
            "extra": "mean: 353.24991979998686 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "t.chaigneau.tc@gmail.com",
            "name": "Thomas Chaigneau",
            "username": "chainyo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "de8e9fab1e171412ff6b9dbb169125321b50fbf3",
          "message": "Add NumPy for all backends dependencies (#332)\n\n* Add NumPy for all backends dependencies\r\n\r\n* numpy as extras",
          "timestamp": "2023-08-21T09:13:55+02:00",
          "tree_id": "00cc9b0c037f51ff436ffe49d5e091ac2c74c102",
          "url": "https://github.com/huggingface/safetensors/commit/de8e9fab1e171412ff6b9dbb169125321b50fbf3"
        },
        "date": 1692602451773,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0293116461958771,
            "unit": "iter/sec",
            "range": "stddev: 0.0157587649606329",
            "extra": "mean: 971.5230598000062 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.6254234996478845,
            "unit": "iter/sec",
            "range": "stddev: 0.10018570688965082",
            "extra": "mean: 380.89093060000323 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.600662808564422,
            "unit": "iter/sec",
            "range": "stddev: 0.0204839259743881",
            "extra": "mean: 384.5173609999847 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.1082427155764325,
            "unit": "iter/sec",
            "range": "stddev: 0.01747688278037413",
            "extra": "mean: 474.32868740000913 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 2.9541753041681704,
            "unit": "iter/sec",
            "range": "stddev: 0.012789298289777125",
            "extra": "mean: 338.50394679999454 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 68.76232756684539,
            "unit": "iter/sec",
            "range": "stddev: 0.0015078853961937274",
            "extra": "mean: 14.542846866663695 msec\nrounds: 15"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3135235708610131,
            "unit": "iter/sec",
            "range": "stddev: 0.11172804825941132",
            "extra": "mean: 761.3110432000099 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.3246446272731247,
            "unit": "iter/sec",
            "range": "stddev: 0.040784836850357144",
            "extra": "mean: 300.78402719998394 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2695e7bb91707b40b437a746cae6c58260a7d386",
          "message": "Reworked the release script entirely (#334)\n\n* Rework the release script entirely.\r\n\r\n* New release script.\r\n\r\n* Ok.",
          "timestamp": "2023-08-22T11:25:11+02:00",
          "tree_id": "4a6b2ac607227e0fcbd387407ff6f8c247bc4c9e",
          "url": "https://github.com/huggingface/safetensors/commit/2695e7bb91707b40b437a746cae6c58260a7d386"
        },
        "date": 1692696621311,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.4951950914422214,
            "unit": "iter/sec",
            "range": "stddev: 0.051901563280257655",
            "extra": "mean: 668.8090441999975 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 4.07285059307437,
            "unit": "iter/sec",
            "range": "stddev: 0.05187277259547482",
            "extra": "mean: 245.52827980000984 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 4.5179363741174585,
            "unit": "iter/sec",
            "range": "stddev: 0.018397859869552783",
            "extra": "mean: 221.33999179998227 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 3.157058185545896,
            "unit": "iter/sec",
            "range": "stddev: 0.009986928850941806",
            "extra": "mean: 316.7505763999998 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 5.249594236288065,
            "unit": "iter/sec",
            "range": "stddev: 0.045924164203189954",
            "extra": "mean: 190.4909132000057 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 126.5828873609214,
            "unit": "iter/sec",
            "range": "stddev: 0.00013583394017370156",
            "extra": "mean: 7.899961999987681 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.8166742725065501,
            "unit": "iter/sec",
            "range": "stddev: 0.02104243233859581",
            "extra": "mean: 550.4564110000047 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 6.197523800790778,
            "unit": "iter/sec",
            "range": "stddev: 0.009175637953173348",
            "extra": "mean: 161.35476557143744 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e8d1f09f74d49ef68d41af09eb79e8e28174d27f",
          "message": "Preparing small patch release (Big Endian fix). (#335)",
          "timestamp": "2023-08-23T13:00:06+02:00",
          "tree_id": "444de72e8c1f1df2068856179a0244b45960570f",
          "url": "https://github.com/huggingface/safetensors/commit/e8d1f09f74d49ef68d41af09eb79e8e28174d27f"
        },
        "date": 1692788722832,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.240097640833815,
            "unit": "iter/sec",
            "range": "stddev: 0.04092866720131954",
            "extra": "mean: 806.3881157999958 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.4296771580549477,
            "unit": "iter/sec",
            "range": "stddev: 0.06010801812046714",
            "extra": "mean: 291.57263320000766 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 3.232688459506295,
            "unit": "iter/sec",
            "range": "stddev: 0.018322627436870707",
            "extra": "mean: 309.3400469999892 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.5588378536356013,
            "unit": "iter/sec",
            "range": "stddev: 0.014620436177134809",
            "extra": "mean: 390.80240999999205 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 6.165721082226908,
            "unit": "iter/sec",
            "range": "stddev: 0.03941971772189453",
            "extra": "mean: 162.1870316000127 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 121.47941273201414,
            "unit": "iter/sec",
            "range": "stddev: 0.00013754237577850794",
            "extra": "mean: 8.231847500004127 msec\nrounds: 6"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3619352924652866,
            "unit": "iter/sec",
            "range": "stddev: 0.04177645169205829",
            "extra": "mean: 734.249274199999 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.743980074094512,
            "unit": "iter/sec",
            "range": "stddev: 0.017771596718508153",
            "extra": "mean: 267.0954386000176 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f10df45ee02ab18e992d990e26368924029dcdc2",
          "message": "Temporary revert of the breaking change (keep it for 0.4.0). (#336)",
          "timestamp": "2023-08-23T13:22:28+02:00",
          "tree_id": "2b71c74c516b1e536cdc38d29a744cccd0f263ff",
          "url": "https://github.com/huggingface/safetensors/commit/f10df45ee02ab18e992d990e26368924029dcdc2"
        },
        "date": 1692790136171,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0696053056456758,
            "unit": "iter/sec",
            "range": "stddev: 0.03841641854384048",
            "extra": "mean: 934.9243078000086 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.609960836817386,
            "unit": "iter/sec",
            "range": "stddev: 0.09426209339399681",
            "extra": "mean: 383.1475115999865 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.595012982730617,
            "unit": "iter/sec",
            "range": "stddev: 0.022246147850812453",
            "extra": "mean: 385.3545267999948 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.0996929395186985,
            "unit": "iter/sec",
            "range": "stddev: 0.013328923293619205",
            "extra": "mean: 476.26011460000655 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.06637739428405,
            "unit": "iter/sec",
            "range": "stddev: 0.06339190012743008",
            "extra": "mean: 245.91913219999242 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 63.74136385794093,
            "unit": "iter/sec",
            "range": "stddev: 0.0003177349677110469",
            "extra": "mean: 15.688399799989837 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.152623999134229,
            "unit": "iter/sec",
            "range": "stddev: 0.047328513015161354",
            "extra": "mean: 867.5856139999951 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.8482392758145916,
            "unit": "iter/sec",
            "range": "stddev: 0.040386606786050216",
            "extra": "mean: 351.0940982000193 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fab53d1e038a6988b8a73106e8bb894737524c96",
          "message": "Don't release rust versions on RC (#337)",
          "timestamp": "2023-08-23T13:25:47+02:00",
          "tree_id": "ef3bcacb3acab25ba611ae4d2231a5b55b4c4570",
          "url": "https://github.com/huggingface/safetensors/commit/fab53d1e038a6988b8a73106e8bb894737524c96"
        },
        "date": 1692790326636,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.0462721642632697,
            "unit": "iter/sec",
            "range": "stddev: 0.06831279390989548",
            "extra": "mean: 955.7742566000002 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 2.80477413709803,
            "unit": "iter/sec",
            "range": "stddev: 0.07671633109230869",
            "extra": "mean: 356.534947599971 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.616197836976742,
            "unit": "iter/sec",
            "range": "stddev: 0.024269194044838648",
            "extra": "mean: 382.2340901999951 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.162656154911937,
            "unit": "iter/sec",
            "range": "stddev: 0.03403610094630705",
            "extra": "mean: 462.3943560000271 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 3.3736663944817056,
            "unit": "iter/sec",
            "range": "stddev: 0.0282222654394272",
            "extra": "mean: 296.41342180000265 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 65.1430294035137,
            "unit": "iter/sec",
            "range": "stddev: 0.0013915129797857047",
            "extra": "mean: 15.350836599964168 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.3924047057427673,
            "unit": "iter/sec",
            "range": "stddev: 0.04857508735482071",
            "extra": "mean: 718.1820025999968 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 2.5202263525572577,
            "unit": "iter/sec",
            "range": "stddev: 0.012465546507016317",
            "extra": "mean: 396.7897561999962 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1a4d0aca8d03e794b7f83623b6395dc226ec7777",
          "message": "Fixing release script. (#338)",
          "timestamp": "2023-08-23T15:20:09+02:00",
          "tree_id": "5287e78c2f202a352df66772411999c5866e9dff",
          "url": "https://github.com/huggingface/safetensors/commit/1a4d0aca8d03e794b7f83623b6395dc226ec7777"
        },
        "date": 1692797118151,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.317320898054514,
            "unit": "iter/sec",
            "range": "stddev: 0.07787669746489027",
            "extra": "mean: 759.1164775999914 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.5326182358331675,
            "unit": "iter/sec",
            "range": "stddev: 0.05429159450932459",
            "extra": "mean: 283.0761586000108 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 3.2743335943490597,
            "unit": "iter/sec",
            "range": "stddev: 0.018144483671937148",
            "extra": "mean: 305.4056562000369 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.6039343252017226,
            "unit": "iter/sec",
            "range": "stddev: 0.026890706467398613",
            "extra": "mean: 384.03426319998744 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.962666933557227,
            "unit": "iter/sec",
            "range": "stddev: 0.05452517549698119",
            "extra": "mean: 201.50455660001398 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 122.0624800389897,
            "unit": "iter/sec",
            "range": "stddev: 0.00006463940693970112",
            "extra": "mean: 8.192525661289006 msec\nrounds: 62"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.6654818029404723,
            "unit": "iter/sec",
            "range": "stddev: 0.04010688995439577",
            "extra": "mean: 600.426854400007 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 4.321568484863132,
            "unit": "iter/sec",
            "range": "stddev: 0.00624388386492602",
            "extra": "mean: 231.39746679999007 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patry.nicolas@protonmail.com",
            "name": "Nicolas Patry",
            "username": "Narsil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d19ecfec62d3f71260ae4468315722380fb1ab4d",
          "message": "Python 3.8 for arm (#339)",
          "timestamp": "2023-08-23T16:28:00+02:00",
          "tree_id": "b27ac52e1985bc1ba611d42f65659c77941db2b5",
          "url": "https://github.com/huggingface/safetensors/commit/d19ecfec62d3f71260ae4468315722380fb1ab4d"
        },
        "date": 1692801200947,
        "tool": "pytest",
        "benches": [
          {
            "name": "benches/test_flax.py::test_flax_flax_load",
            "value": 1.1490778849223517,
            "unit": "iter/sec",
            "range": "stddev: 0.03764085848353429",
            "extra": "mean: 870.2630284000065 msec\nrounds: 5"
          },
          {
            "name": "benches/test_flax.py::test_flax_sf_load",
            "value": 3.22718705056146,
            "unit": "iter/sec",
            "range": "stddev: 0.05023764387262277",
            "extra": "mean: 309.8673811999902 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_paddle_load",
            "value": 2.9121503316302606,
            "unit": "iter/sec",
            "range": "stddev: 0.01835573621904825",
            "extra": "mean: 343.3888659999866 msec\nrounds: 5"
          },
          {
            "name": "benches/test_paddle.py::test_paddle_sf_load",
            "value": 2.4988075927553566,
            "unit": "iter/sec",
            "range": "stddev: 0.03278398434892794",
            "extra": "mean: 400.19087620000846 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_pt_load_cpu",
            "value": 4.360483851148589,
            "unit": "iter/sec",
            "range": "stddev: 0.07139388740385637",
            "extra": "mean: 229.33234800000264 msec\nrounds: 5"
          },
          {
            "name": "benches/test_pt.py::test_pt_sf_load_cpu",
            "value": 79.76703304143469,
            "unit": "iter/sec",
            "range": "stddev: 0.000517641294633844",
            "extra": "mean: 12.536507400000119 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_tf_load",
            "value": 1.5494791850586653,
            "unit": "iter/sec",
            "range": "stddev: 0.03605531231676572",
            "extra": "mean: 645.3781435999986 msec\nrounds: 5"
          },
          {
            "name": "benches/test_tf.py::test_tf_sf_load",
            "value": 3.750894457046682,
            "unit": "iter/sec",
            "range": "stddev: 0.012415770094241765",
            "extra": "mean: 266.60307600000124 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}