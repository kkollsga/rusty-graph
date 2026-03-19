window.BENCHMARK_DATA = {
  "lastUpdate": 1773911681241,
  "repoUrl": "https://github.com/kkollsga/kglite",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "fd42d169df8b120bb82b8ab27209f971d87ae797",
          "message": "fix: add Python 3.13 to CI matrix, grant benchmark write permission\n\n- Add Python 3.13 to the test matrix\n- Add permissions: contents: write to benchmark job so\n  github-actions[bot] can push to gh-pages\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T09:25:17+01:00",
          "tree_id": "fca2a2ba0e1a945f05d0171e75d83aabfedfe7dc",
          "url": "https://github.com/kkollsga/kglite/commit/fd42d169df8b120bb82b8ab27209f971d87ae797"
        },
        "date": 1773044990446,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1165.0952683638106,
            "unit": "iter/sec",
            "range": "stddev: 0.000025658180919542606",
            "extra": "mean: 858.2989109588777 usec\nrounds: 438"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 812.43958628117,
            "unit": "iter/sec",
            "range": "stddev: 0.00011224763438778204",
            "extra": "mean: 1.2308607518466226 msec\nrounds: 677"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13151.421204225588,
            "unit": "iter/sec",
            "range": "stddev: 0.000004140427148919118",
            "extra": "mean: 76.0374095294505 usec\nrounds: 4974"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1596.6443742319068,
            "unit": "iter/sec",
            "range": "stddev: 0.000019286200600634077",
            "extra": "mean: 626.3135461715244 usec\nrounds: 888"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 651918.7042620395,
            "unit": "iter/sec",
            "range": "stddev: 3.989128339401837e-7",
            "extra": "mean: 1.5339335924284954 usec\nrounds: 44498"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 134903.28707557428,
            "unit": "iter/sec",
            "range": "stddev: 9.684000201294694e-7",
            "extra": "mean: 7.412717819394491 usec\nrounds: 18031"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "79dcae3eadacc451b83ad2053fd8c00932bc901c",
          "message": ".pyi bug fix",
          "timestamp": "2026-03-12T22:50:11+01:00",
          "tree_id": "42a71cd00dbf37b6f88df9fbab94845b0719e354",
          "url": "https://github.com/kkollsga/kglite/commit/79dcae3eadacc451b83ad2053fd8c00932bc901c"
        },
        "date": 1773352334441,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1075.0009616424525,
            "unit": "iter/sec",
            "range": "stddev: 0.000020059316788602697",
            "extra": "mean: 930.2317259997039 usec\nrounds: 500"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 806.5886354464182,
            "unit": "iter/sec",
            "range": "stddev: 0.000025917633712934622",
            "extra": "mean: 1.23978934992871 msec\nrounds: 703"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13641.804608353508,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035701586827171303",
            "extra": "mean: 73.3040846654301 usec\nrounds: 5693"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1759.2773919224915,
            "unit": "iter/sec",
            "range": "stddev: 0.000013136315460030678",
            "extra": "mean: 568.4151939832676 usec\nrounds: 964"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 687718.074760811,
            "unit": "iter/sec",
            "range": "stddev: 3.538571303625668e-7",
            "extra": "mean: 1.4540842195369095 usec\nrounds: 55854"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 138019.92515121287,
            "unit": "iter/sec",
            "range": "stddev: 9.200685857418458e-7",
            "extra": "mean: 7.245330693408309 usec\nrounds: 32450"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "8a9f05f69992528729839f460a0b964c550fcae9",
          "message": "feat: columnar property storage and memory-mapped directory format\n\nAdd per-type columnar storage (enable_columnar/disable_columnar) that moves\nnode properties from per-node maps into typed column stores, reducing memory\nfor homogeneous columns. Add save_mmap/load_mmap directory format with\nmemory-mapped column files for out-of-core workloads.\n\nPhase A: Cow<Value> return types preserving zero-copy for Map/Compact.\nPhase B: ColumnStore + TypedColumn + PropertyStorage::Columnar variant.\nPhase C: MmapOrVec/MmapBytes file-backed columns, directory save/load.\nPhase D: Python API, type stubs, introspection, benchmarks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-18T00:53:07+01:00",
          "tree_id": "8b6d4ac5cb7986770b00ed4bf35719f9ce7b4c7e",
          "url": "https://github.com/kkollsga/kglite/commit/8a9f05f69992528729839f460a0b964c550fcae9"
        },
        "date": 1773791728045,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1066.253369871664,
            "unit": "iter/sec",
            "range": "stddev: 0.000039564991072661136",
            "extra": "mean: 937.8633899373857 usec\nrounds: 477"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 797.949248832039,
            "unit": "iter/sec",
            "range": "stddev: 0.00002999881808443218",
            "extra": "mean: 1.2532125338343303 msec\nrounds: 665"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13763.857438922449,
            "unit": "iter/sec",
            "range": "stddev: 0.000004534976845773097",
            "extra": "mean: 72.65405097644549 usec\nrounds: 5787"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1708.2486137543112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254764414252603",
            "extra": "mean: 585.3948845316171 usec\nrounds: 918"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 688654.999997676,
            "unit": "iter/sec",
            "range": "stddev: 4.2230426950218746e-7",
            "extra": "mean: 1.4521059166104575 usec\nrounds: 94886"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 137493.73195631197,
            "unit": "iter/sec",
            "range": "stddev: 9.772901081363317e-7",
            "extra": "mean: 7.273058820730428 usec\nrounds: 21727"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2797.5781248239664,
            "unit": "iter/sec",
            "range": "stddev: 0.00001324612977943654",
            "extra": "mean: 357.4520372198448 usec\nrounds: 5266"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1670.2215702497615,
            "unit": "iter/sec",
            "range": "stddev: 0.000023655346223086946",
            "extra": "mean: 598.7229585655884 usec\nrounds: 1255"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13568.54411177825,
            "unit": "iter/sec",
            "range": "stddev: 0.00001706253399889654",
            "extra": "mean: 73.69987463370845 usec\nrounds: 10577"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1067.3189996345175,
            "unit": "iter/sec",
            "range": "stddev: 0.000058403886503522734",
            "extra": "mean: 936.9270108959275 usec\nrounds: 826"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_mmap",
            "value": 636.0986666859234,
            "unit": "iter/sec",
            "range": "stddev: 0.00003919815005790763",
            "extra": "mean: 1.5720831568630758 msec\nrounds: 663"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "1a059fd20c796adfb295f615116a0ce3adcca9a0",
          "message": "style: ruff format Python test files\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-18T00:55:41+01:00",
          "tree_id": "89cfebb4697a3a69ea680f46224920df2686b677",
          "url": "https://github.com/kkollsga/kglite/commit/1a059fd20c796adfb295f615116a0ce3adcca9a0"
        },
        "date": 1773791868745,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1049.6351809669752,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199522882488931",
            "extra": "mean: 952.7119690088429 usec\nrounds: 484"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 787.7758598153356,
            "unit": "iter/sec",
            "range": "stddev: 0.00002980464872439012",
            "extra": "mean: 1.2693966025239873 msec\nrounds: 634"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13671.054846923584,
            "unit": "iter/sec",
            "range": "stddev: 0.000004482469372942293",
            "extra": "mean: 73.1472451246168 usec\nrounds: 5589"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1723.2902616358408,
            "unit": "iter/sec",
            "range": "stddev: 0.00010122318366399527",
            "extra": "mean: 580.285296251106 usec\nrounds: 827"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 695566.3728808132,
            "unit": "iter/sec",
            "range": "stddev: 4.077897344098187e-7",
            "extra": "mean: 1.4376773216599303 usec\nrounds: 116878"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130956.08217863101,
            "unit": "iter/sec",
            "range": "stddev: 0.000001127968631771582",
            "extra": "mean: 7.636147808972684 usec\nrounds: 21406"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2873.714587726904,
            "unit": "iter/sec",
            "range": "stddev: 0.000011363566317003201",
            "extra": "mean: 347.98166953350636 usec\nrounds: 5232"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1721.4722495889655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000149162818444247",
            "extra": "mean: 580.8981238232386 usec\nrounds: 1381"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14545.919569444919,
            "unit": "iter/sec",
            "range": "stddev: 0.000004897287190903247",
            "extra": "mean: 68.74780210531307 usec\nrounds: 11208"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1065.890436326334,
            "unit": "iter/sec",
            "range": "stddev: 0.00007071563122585359",
            "extra": "mean: 938.182730531451 usec\nrounds: 809"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_mmap",
            "value": 637.5039002154459,
            "unit": "iter/sec",
            "range": "stddev: 0.00003702622875036059",
            "extra": "mean: 1.5686178541998688 msec\nrounds: 631"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "e569fa6ea6313afaee2d3e116c3f933af69ee026",
          "message": "fix: ruff lint errors in test files\n\nFix import sorting (I001), remove unused import (F401), remove\nextraneous f-prefix (F541), and fix line length (E501).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-18T07:16:59+01:00",
          "tree_id": "60fc2ff1b82c006ac808965f5ca022d3d8c2868c",
          "url": "https://github.com/kkollsga/kglite/commit/e569fa6ea6313afaee2d3e116c3f933af69ee026"
        },
        "date": 1773814755929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1072.9610229058705,
            "unit": "iter/sec",
            "range": "stddev: 0.00004460577153821641",
            "extra": "mean: 932.0003044395105 usec\nrounds: 473"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 803.6605061966195,
            "unit": "iter/sec",
            "range": "stddev: 0.00002786185049330185",
            "extra": "mean: 1.2443065103853008 msec\nrounds: 674"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13480.1945126787,
            "unit": "iter/sec",
            "range": "stddev: 0.000003722434681922909",
            "extra": "mean: 74.18290582227557 usec\nrounds: 5702"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1702.7490967223564,
            "unit": "iter/sec",
            "range": "stddev: 0.00002508216920686688",
            "extra": "mean: 587.2855853659902 usec\nrounds: 861"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 677123.6532043569,
            "unit": "iter/sec",
            "range": "stddev: 3.7821689045668745e-7",
            "extra": "mean: 1.4768351323538813 usec\nrounds: 96071"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131503.14452686417,
            "unit": "iter/sec",
            "range": "stddev: 9.5371731626179e-7",
            "extra": "mean: 7.604380895969484 usec\nrounds: 21116"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2786.3445293859645,
            "unit": "iter/sec",
            "range": "stddev: 0.000020668797560565272",
            "extra": "mean: 358.89316251223715 usec\nrounds: 5095"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1670.2264399258565,
            "unit": "iter/sec",
            "range": "stddev: 0.000015011409744515607",
            "extra": "mean: 598.7212129418758 usec\nrounds: 1329"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13636.59966396779,
            "unit": "iter/sec",
            "range": "stddev: 0.000004117678154351599",
            "extra": "mean: 73.33206405129839 usec\nrounds: 10757"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1071.758224247485,
            "unit": "iter/sec",
            "range": "stddev: 0.0000455515664767029",
            "extra": "mean: 933.0462574263252 usec\nrounds: 808"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_mmap",
            "value": 639.6131150762404,
            "unit": "iter/sec",
            "range": "stddev: 0.00011244751151591745",
            "extra": "mean: 1.563445114599944 msec\nrounds: 637"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "0b9b56ee344037326342fac8ef6601db9de9e749",
          "message": "feat: memory management API and vacuum columnar rebuild (0.6.5)\n\nAdd set_memory_limit(), unspill(), automatic spill-to-disk, and vacuum\ncolumnar rebuild. Deleting nodes no longer leaks orphaned columnar rows —\nvacuum() rebuilds column stores from live nodes only.\n\nIncludes 29 new tests and 15 memory benchmarks (heap vs mmap).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-18T10:10:29+01:00",
          "tree_id": "59fbc0ca771febd6c6b791fc711e92ea6ca901be",
          "url": "https://github.com/kkollsga/kglite/commit/0b9b56ee344037326342fac8ef6601db9de9e749"
        },
        "date": 1773825179514,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1062.2850193532431,
            "unit": "iter/sec",
            "range": "stddev: 0.00002241245366349875",
            "extra": "mean: 941.3669418108103 usec\nrounds: 464"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 796.4421013967849,
            "unit": "iter/sec",
            "range": "stddev: 0.000034451393157775907",
            "extra": "mean: 1.2555840509262621 msec\nrounds: 648"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13293.257314742737,
            "unit": "iter/sec",
            "range": "stddev: 0.000004789195638742319",
            "extra": "mean: 75.22610721534453 usec\nrounds: 5419"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1643.7987175479736,
            "unit": "iter/sec",
            "range": "stddev: 0.000025801375042236432",
            "extra": "mean: 608.3469887917195 usec\nrounds: 803"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 688830.4522252987,
            "unit": "iter/sec",
            "range": "stddev: 3.9760209924229627e-7",
            "extra": "mean: 1.4517360502420498 usec\nrounds: 84876"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 132356.51168052558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010441143515236963",
            "extra": "mean: 7.5553517337608715 usec\nrounds: 17735"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2827.5107474891247,
            "unit": "iter/sec",
            "range": "stddev: 0.00006107304169525008",
            "extra": "mean: 353.6679748743718 usec\nrounds: 2388"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1661.4259036734654,
            "unit": "iter/sec",
            "range": "stddev: 0.000020033082814860997",
            "extra": "mean: 601.8926259600072 usec\nrounds: 1302"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14340.29804580876,
            "unit": "iter/sec",
            "range": "stddev: 0.000005001754981278796",
            "extra": "mean: 69.73355761544093 usec\nrounds: 8253"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1072.48309457215,
            "unit": "iter/sec",
            "range": "stddev: 0.00033937693055565614",
            "extra": "mean: 932.4156297297478 usec\nrounds: 740"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_mmap",
            "value": 898.8234414516376,
            "unit": "iter/sec",
            "range": "stddev: 0.000023263914626234657",
            "extra": "mean: 1.1125655539033985 msec\nrounds: 807"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "3b354eea70d1db37953988caed3ad54a63ae3458",
          "message": "feat: v3 unified columnar file format with temp dir cleanup (0.6.6)\n\nReplace v1/v2/mmap formats with a single v3 .kgl file that separates\ntopology from per-type columnar sections (zstd-compressed). Loaded\ngraphs use memory-mapped temp files for larger-than-RAM support.\n\nKey changes:\n- v3 file format: magic b\"RGF\\x03\", stripped topology + packed columns\n- StripPropertiesGuard for zero-property topology serialization\n- Temp dir cleanup via Drop impl on DirGraph (fixes leak)\n- save() auto-enables columnar, stays columnar (no disable step)\n- Removed save_mmap/load_mmap, v1/v2 format support\n- Extracted zstd/bincode helpers, metadata transfer helpers\n- Eliminated double buffering in column packing\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-18T14:20:29+01:00",
          "tree_id": "2da0afa3a845b5c6a2917ac6168e72baec881dd9",
          "url": "https://github.com/kkollsga/kglite/commit/3b354eea70d1db37953988caed3ad54a63ae3458"
        },
        "date": 1773840165555,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1101.3481817317424,
            "unit": "iter/sec",
            "range": "stddev: 0.00002056349199709084",
            "extra": "mean: 907.9780732262307 usec\nrounds: 437"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 802.2126876074095,
            "unit": "iter/sec",
            "range": "stddev: 0.00013128762800941891",
            "extra": "mean: 1.2465522117114463 msec\nrounds: 666"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 14444.550561180231,
            "unit": "iter/sec",
            "range": "stddev: 0.000004817296429905994",
            "extra": "mean: 69.23026062766554 usec\nrounds: 5928"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1732.2687191915606,
            "unit": "iter/sec",
            "range": "stddev: 0.000018170894874734658",
            "extra": "mean: 577.277641119499 usec\nrounds: 822"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 699739.776120289,
            "unit": "iter/sec",
            "range": "stddev: 4.2632281285469257e-7",
            "extra": "mean: 1.429102695211219 usec\nrounds: 67491"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 139546.60615321182,
            "unit": "iter/sec",
            "range": "stddev: 9.205252742538655e-7",
            "extra": "mean: 7.166064640096472 usec\nrounds: 20560"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2849.275921046513,
            "unit": "iter/sec",
            "range": "stddev: 0.000020535197019987336",
            "extra": "mean: 350.9663604754394 usec\nrounds: 4458"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1689.8248650784988,
            "unit": "iter/sec",
            "range": "stddev: 0.000025516190140046365",
            "extra": "mean: 591.777302290759 usec\nrounds: 1353"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14404.707976558664,
            "unit": "iter/sec",
            "range": "stddev: 0.0000039865235179666005",
            "extra": "mean: 69.4217475027844 usec\nrounds: 11513"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1320.6289614826972,
            "unit": "iter/sec",
            "range": "stddev: 0.00028020348396644935",
            "extra": "mean: 757.2149552719785 usec\nrounds: 939"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1338.9915065872563,
            "unit": "iter/sec",
            "range": "stddev: 0.000017305102997708155",
            "extra": "mean: 746.8307267674476 usec\nrounds: 1259"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "db615b9c14ddfc272668b84e34b2f3c799215cc8",
          "message": "doc uodates",
          "timestamp": "2026-03-19T07:58:17+01:00",
          "tree_id": "385abb984fe4b1122a3f275af9d65485589763aa",
          "url": "https://github.com/kkollsga/kglite/commit/db615b9c14ddfc272668b84e34b2f3c799215cc8"
        },
        "date": 1773903638146,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1055.7124192696594,
            "unit": "iter/sec",
            "range": "stddev: 0.00003757214054594923",
            "extra": "mean: 947.2276557017286 usec\nrounds: 456"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 786.5273934881802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000398388170416655",
            "extra": "mean: 1.2714115341426158 msec\nrounds: 659"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13492.882277829887,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051822000142508306",
            "extra": "mean: 74.11314939307644 usec\nrounds: 5355"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1715.52506608492,
            "unit": "iter/sec",
            "range": "stddev: 0.000022223416004653862",
            "extra": "mean: 582.91191412443 usec\nrounds: 885"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 706403.6595322631,
            "unit": "iter/sec",
            "range": "stddev: 3.9203991673442433e-7",
            "extra": "mean: 1.415621205391459 usec\nrounds: 108378"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131915.09291144388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015803957393744693",
            "extra": "mean: 7.580633708618252 usec\nrounds: 19722"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2715.0828233382736,
            "unit": "iter/sec",
            "range": "stddev: 0.000013759718495312256",
            "extra": "mean: 368.31288953847485 usec\nrounds: 4789"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1670.7887192595213,
            "unit": "iter/sec",
            "range": "stddev: 0.000050900824664409125",
            "extra": "mean: 598.5197221364956 usec\nrounds: 1292"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13731.094360810379,
            "unit": "iter/sec",
            "range": "stddev: 0.000005021972386545394",
            "extra": "mean: 72.82740717696024 usec\nrounds: 10450"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1310.744856558437,
            "unit": "iter/sec",
            "range": "stddev: 0.00007753557781645549",
            "extra": "mean: 762.9249849781249 usec\nrounds: 932"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1328.9084484907717,
            "unit": "iter/sec",
            "range": "stddev: 0.000018828850056274787",
            "extra": "mean: 752.4972853740904 usec\nrounds: 1258"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "kkollsg@gmail.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "distinct": true,
          "id": "f4fa1cebcf8f83bdf7bff69b20396adb60ef0935",
          "message": "fixes",
          "timestamp": "2026-03-19T10:12:28+01:00",
          "tree_id": "17f53cef0a42038088e414b11698ffa0db020129",
          "url": "https://github.com/kkollsga/kglite/commit/f4fa1cebcf8f83bdf7bff69b20396adb60ef0935"
        },
        "date": 1773911680249,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1036.4504614317248,
            "unit": "iter/sec",
            "range": "stddev: 0.00002171873617557585",
            "extra": "mean: 964.8314484983943 usec\nrounds: 466"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 783.2453342436344,
            "unit": "iter/sec",
            "range": "stddev: 0.00003153076084123452",
            "extra": "mean: 1.276739172619115 msec\nrounds: 672"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13617.223176839932,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042225505348227955",
            "extra": "mean: 73.43641115472002 usec\nrounds: 5307"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1717.830242975059,
            "unit": "iter/sec",
            "range": "stddev: 0.000013954199140255452",
            "extra": "mean: 582.129697674975 usec\nrounds: 946"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 701445.849643122,
            "unit": "iter/sec",
            "range": "stddev: 3.427603375443272e-7",
            "extra": "mean: 1.4256267971487389 usec\nrounds: 126663"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 134301.07215461,
            "unit": "iter/sec",
            "range": "stddev: 9.724906235046944e-7",
            "extra": "mean: 7.445956938071057 usec\nrounds: 22456"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2750.4119139755057,
            "unit": "iter/sec",
            "range": "stddev: 0.000013431031368449385",
            "extra": "mean: 363.5819038300259 usec\nrounds: 4804"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1656.2909639475874,
            "unit": "iter/sec",
            "range": "stddev: 0.000046498226494765",
            "extra": "mean: 603.758652173414 usec\nrounds: 1242"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14202.372924948826,
            "unit": "iter/sec",
            "range": "stddev: 0.000013930462345937704",
            "extra": "mean: 70.41076905136985 usec\nrounds: 10708"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1320.093215821421,
            "unit": "iter/sec",
            "range": "stddev: 0.00009220847049639785",
            "extra": "mean: 757.5222628333526 usec\nrounds: 974"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1357.3701562757528,
            "unit": "iter/sec",
            "range": "stddev: 0.000013651698569397648",
            "extra": "mean: 736.718716981168 usec\nrounds: 1272"
          }
        ]
      }
    ]
  }
}