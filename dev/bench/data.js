window.BENCHMARK_DATA = {
  "lastUpdate": 1673975489447,
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
      }
    ]
  }
}