window.BENCHMARK_DATA = {
  "lastUpdate": 1667404736104,
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
      }
    ]
  }
}