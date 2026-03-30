window.BENCHMARK_DATA = {
  "lastUpdate": 1774892274531,
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
          "id": "3ae7325e4758db913296b069e13595309316133d",
          "message": "feat: poincaré distance metric, embedding_norm(), and stored metric (0.6.9)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-22T01:29:21+01:00",
          "tree_id": "e415ea8c6f78feb7a2dddb37a2d26e6e54101520",
          "url": "https://github.com/kkollsga/kglite/commit/3ae7325e4758db913296b069e13595309316133d"
        },
        "date": 1774139484497,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1074.0769838486653,
            "unit": "iter/sec",
            "range": "stddev: 0.000017240781914447274",
            "extra": "mean: 931.0319604994883 usec\nrounds: 481"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 798.5328361522804,
            "unit": "iter/sec",
            "range": "stddev: 0.00005530348641806349",
            "extra": "mean: 1.2522966554744153 msec\nrounds: 685"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13850.031459735199,
            "unit": "iter/sec",
            "range": "stddev: 0.000005724190996660747",
            "extra": "mean: 72.20200206094833 usec\nrounds: 6308"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1711.9477219553094,
            "unit": "iter/sec",
            "range": "stddev: 0.000024649915159988674",
            "extra": "mean: 584.1299866667921 usec\nrounds: 900"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 693392.4499881783,
            "unit": "iter/sec",
            "range": "stddev: 4.1351039766618274e-7",
            "extra": "mean: 1.4421847252837106 usec\nrounds: 119675"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 132877.94004002627,
            "unit": "iter/sec",
            "range": "stddev: 9.512892012773435e-7",
            "extra": "mean: 7.525703662314259 usec\nrounds: 21489"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2818.4871969124665,
            "unit": "iter/sec",
            "range": "stddev: 0.00003701806384575468",
            "extra": "mean: 354.8002634517757 usec\nrounds: 4832"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1687.1915062022701,
            "unit": "iter/sec",
            "range": "stddev: 0.00005071082822288719",
            "extra": "mean: 592.7009449276555 usec\nrounds: 1380"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14283.34748884616,
            "unit": "iter/sec",
            "range": "stddev: 0.000004913120680763287",
            "extra": "mean: 70.0115992263647 usec\nrounds: 11892"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1269.6802310054597,
            "unit": "iter/sec",
            "range": "stddev: 0.00025469387473757806",
            "extra": "mean: 787.5998819073524 usec\nrounds: 923"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1314.7024877144588,
            "unit": "iter/sec",
            "range": "stddev: 0.000013736190974575924",
            "extra": "mean: 760.6283621920024 usec\nrounds: 1259"
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
          "id": "6224f45f6186d7322286699bc3272a602a3c030c",
          "message": "fix: multi-MATCH empty propagation and test suite consolidation (0.6.10)\n\nFix multi-MATCH queries returning incorrect results when first MATCH\nproduces 0 rows. Guard executor loops and restrict planner fusion to\nfirst-clause position. Migrate unique tests from untracked pytest/ into\ntests/ suite (1609 tests passing).\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-29T15:14:04+02:00",
          "tree_id": "b9084defb17ca0f58479c46780290bfdb5560ae9",
          "url": "https://github.com/kkollsga/kglite/commit/6224f45f6186d7322286699bc3272a602a3c030c"
        },
        "date": 1774790215760,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1048.3569291819874,
            "unit": "iter/sec",
            "range": "stddev: 0.00003401978751271203",
            "extra": "mean: 953.8736017896888 usec\nrounds: 447"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 782.1685361782687,
            "unit": "iter/sec",
            "range": "stddev: 0.00004122428506376773",
            "extra": "mean: 1.2784968376330648 msec\nrounds: 659"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13466.039145323655,
            "unit": "iter/sec",
            "range": "stddev: 0.000005508092410178307",
            "extra": "mean: 74.26088616022399 usec\nrounds: 6597"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1747.7633359080285,
            "unit": "iter/sec",
            "range": "stddev: 0.00005112241044395311",
            "extra": "mean: 572.1598453605634 usec\nrounds: 970"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 704261.3394344178,
            "unit": "iter/sec",
            "range": "stddev: 4.4868270897109505e-7",
            "extra": "mean: 1.4199274388724585 usec\nrounds: 130993"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131739.30659813594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012777143485760676",
            "extra": "mean: 7.590748925455099 usec\nrounds: 19078"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2734.3901528335546,
            "unit": "iter/sec",
            "range": "stddev: 0.000011461168138043548",
            "extra": "mean: 365.7122590804148 usec\nrounds: 4350"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1720.1274843813287,
            "unit": "iter/sec",
            "range": "stddev: 0.00003146979018792941",
            "extra": "mean: 581.3522596900228 usec\nrounds: 1290"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13641.410145017137,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050292799424877925",
            "extra": "mean: 73.30620437105432 usec\nrounds: 9517"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1271.309346523494,
            "unit": "iter/sec",
            "range": "stddev: 0.00011328506398746277",
            "extra": "mean: 786.5906144201543 usec\nrounds: 957"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1306.3236071543142,
            "unit": "iter/sec",
            "range": "stddev: 0.00002395363975825732",
            "extra": "mean: 765.5071029286476 usec\nrounds: 1195"
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
          "id": "d1bd005d995824d60a42c62c696c0fe933e67c16",
          "message": "fix: 19 Cypher engine bugs, performance benchmarks, and code structure improvements (0.6.11)\n\nSystematic resolution of BUG-01 through BUG-20 (except BUG-04) discovered\nvia legal knowledge graph testing. Fixes silent wrong results (equality+GROUP BY,\nint-to-float conversion, HAVING propagation, RETURN *, multi-hop paths),\nerrors on valid syntax (stDev, datetime, date().year, pipe types, XOR, modulo,\nhead/last, IN with variables), and less common patterns (boolean RETURN\nexpressions, null comparisons, map {.*} projection).\n\nKey structural improvements: PredicateExpr AST variant bridging expression/\npredicate boundary, ExprPropertyAccess for function result properties,\nmulti-type edge matching, virtual type property in pattern matcher, and\ncentralized aggregate sum integer preservation.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-29T23:02:33+02:00",
          "tree_id": "f51953d30f4823981f5a84009b3480b58941c7e9",
          "url": "https://github.com/kkollsga/kglite/commit/d1bd005d995824d60a42c62c696c0fe933e67c16"
        },
        "date": 1774818375824,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1047.1970626701298,
            "unit": "iter/sec",
            "range": "stddev: 0.00013796921575101164",
            "extra": "mean: 954.9301040342997 usec\nrounds: 471"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 799.9775046787408,
            "unit": "iter/sec",
            "range": "stddev: 0.00002707012416205851",
            "extra": "mean: 1.2500351499278537 msec\nrounds: 687"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 14577.514542640885,
            "unit": "iter/sec",
            "range": "stddev: 0.000004073159225429984",
            "extra": "mean: 68.59879968391637 usec\nrounds: 7598"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1614.059323531205,
            "unit": "iter/sec",
            "range": "stddev: 0.000020393076804692268",
            "extra": "mean: 619.5559143465812 usec\nrounds: 934"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 696528.8575117487,
            "unit": "iter/sec",
            "range": "stddev: 4.60411513293052e-7",
            "extra": "mean: 1.4356906956767868 usec\nrounds: 115527"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 135547.90724935677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012611417427016341",
            "extra": "mean: 7.377465431173194 usec\nrounds: 21638"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2844.9445778210147,
            "unit": "iter/sec",
            "range": "stddev: 0.0000147996493984323",
            "extra": "mean: 351.5006962862928 usec\nrounds: 5143"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1586.53819703784,
            "unit": "iter/sec",
            "range": "stddev: 0.000029634905119513733",
            "extra": "mean: 630.3031353843598 usec\nrounds: 1300"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14808.59176261583,
            "unit": "iter/sec",
            "range": "stddev: 0.000004548729509143043",
            "extra": "mean: 67.5283656967634 usec\nrounds: 11824"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1303.1176263912657,
            "unit": "iter/sec",
            "range": "stddev: 0.00021419819747671374",
            "extra": "mean: 767.3904333327976 usec\nrounds: 1020"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1363.4383718132929,
            "unit": "iter/sec",
            "range": "stddev: 0.000016030692232356727",
            "extra": "mean: 733.4398243978264 usec\nrounds: 1287"
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
          "id": "486e6cb6d0b4ca0a5a2268cd91b7886832d37f2d",
          "message": "fix",
          "timestamp": "2026-03-29T23:41:26+02:00",
          "tree_id": "5619e86007b85d0cc436a16ea8b499f0abb7ff50",
          "url": "https://github.com/kkollsga/kglite/commit/486e6cb6d0b4ca0a5a2268cd91b7886832d37f2d"
        },
        "date": 1774820615726,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1052.6851836382662,
            "unit": "iter/sec",
            "range": "stddev: 0.000021003929493632543",
            "extra": "mean: 949.9516242299745 usec\nrounds: 487"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 791.4238807571147,
            "unit": "iter/sec",
            "range": "stddev: 0.00002747562635922776",
            "extra": "mean: 1.2635453949700772 msec\nrounds: 676"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13654.571631029728,
            "unit": "iter/sec",
            "range": "stddev: 0.000005207411250614686",
            "extra": "mean: 73.23554535592467 usec\nrounds: 6471"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1573.1988005699386,
            "unit": "iter/sec",
            "range": "stddev: 0.000022334186816825694",
            "extra": "mean: 635.6475733630866 usec\nrounds: 886"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 685853.103369279,
            "unit": "iter/sec",
            "range": "stddev: 0.000003322861543855901",
            "extra": "mean: 1.4580381645682765 usec\nrounds: 118120"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 133558.58180113966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013405161126457486",
            "extra": "mean: 7.487351142204679 usec\nrounds: 21319"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2837.661030202829,
            "unit": "iter/sec",
            "range": "stddev: 0.000010053753643371324",
            "extra": "mean: 352.4029083658814 usec\nrounds: 4638"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1576.3840159602255,
            "unit": "iter/sec",
            "range": "stddev: 0.00002679868512249535",
            "extra": "mean: 634.3631944218035 usec\nrounds: 1255"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13960.452131489323,
            "unit": "iter/sec",
            "range": "stddev: 0.00000781740919189275",
            "extra": "mean: 71.63091786578967 usec\nrounds: 10982"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1327.6656185567476,
            "unit": "iter/sec",
            "range": "stddev: 0.0001256204229797561",
            "extra": "mean: 753.2016993006569 usec\nrounds: 1001"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1366.3113675216528,
            "unit": "iter/sec",
            "range": "stddev: 0.000014946607338680411",
            "extra": "mean: 731.8975921380911 usec\nrounds: 1221"
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
          "id": "7b40fa7d86d9b42eab463dfab499bfdb8260f588",
          "message": "Merge branch 'main' of https://github.com/kkollsga/kglite",
          "timestamp": "2026-03-29T23:54:18+02:00",
          "tree_id": "9b814ebbe387155dd6c31b0e124ec29c4c9d8aa9",
          "url": "https://github.com/kkollsga/kglite/commit/7b40fa7d86d9b42eab463dfab499bfdb8260f588"
        },
        "date": 1774821381261,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1066.8206258335695,
            "unit": "iter/sec",
            "range": "stddev: 0.000020435795269879898",
            "extra": "mean: 937.3647038541662 usec\nrounds: 493"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 794.3688713251737,
            "unit": "iter/sec",
            "range": "stddev: 0.00010451596544278127",
            "extra": "mean: 1.258861010416723 msec\nrounds: 672"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 14250.613723735092,
            "unit": "iter/sec",
            "range": "stddev: 0.000004863748179808447",
            "extra": "mean: 70.17241638754486 usec\nrounds: 7457"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1608.75284324674,
            "unit": "iter/sec",
            "range": "stddev: 0.000018784273477230216",
            "extra": "mean: 621.599523008039 usec\nrounds: 891"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 677089.7388405282,
            "unit": "iter/sec",
            "range": "stddev: 4.7236894224712717e-7",
            "extra": "mean: 1.4769091047110454 usec\nrounds: 118400"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 132862.69342055914,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012361742265612534",
            "extra": "mean: 7.526567272233698 usec\nrounds: 21138"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2823.501935713843,
            "unit": "iter/sec",
            "range": "stddev: 0.000012352406539561463",
            "extra": "mean: 354.1701131319317 usec\nrounds: 4234"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1602.15664630839,
            "unit": "iter/sec",
            "range": "stddev: 0.00003120546347000557",
            "extra": "mean: 624.1586940354119 usec\nrounds: 1291"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14573.561223217266,
            "unit": "iter/sec",
            "range": "stddev: 0.000005677877174428072",
            "extra": "mean: 68.6174082424611 usec\nrounds: 11841"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1314.2735614878445,
            "unit": "iter/sec",
            "range": "stddev: 0.00021774656407242231",
            "extra": "mean: 760.8766008105146 usec\nrounds: 987"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1371.2066739891457,
            "unit": "iter/sec",
            "range": "stddev: 0.000014938619171076832",
            "extra": "mean: 729.2846650831834 usec\nrounds: 1263"
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
          "id": "d4d8c07ff3c40961b68a9320728faff3f9ca4970",
          "message": "push",
          "timestamp": "2026-03-30T08:17:57+02:00",
          "tree_id": "c5f633e449b0b8c29ce9408beced1e3e288ff839",
          "url": "https://github.com/kkollsga/kglite/commit/d4d8c07ff3c40961b68a9320728faff3f9ca4970"
        },
        "date": 1774851614531,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1075.324140899713,
            "unit": "iter/sec",
            "range": "stddev: 0.00002032663545673595",
            "extra": "mean: 929.952152997616 usec\nrounds: 634"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 795.6480527997446,
            "unit": "iter/sec",
            "range": "stddev: 0.000028003261024619118",
            "extra": "mean: 1.2568371109326253 msec\nrounds: 622"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13865.001214265776,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042251376336337365",
            "extra": "mean: 72.12404705533632 usec\nrounds: 6758"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1618.0725298736263,
            "unit": "iter/sec",
            "range": "stddev: 0.000018318475229216564",
            "extra": "mean: 618.0192677012453 usec\nrounds: 1031"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 699549.2392549756,
            "unit": "iter/sec",
            "range": "stddev: 4.0654384288648245e-7",
            "extra": "mean: 1.4294919412177567 usec\nrounds: 123840"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 135106.64271231712,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012806436221972328",
            "extra": "mean: 7.401560574111092 usec\nrounds: 34206"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2730.120772596002,
            "unit": "iter/sec",
            "range": "stddev: 0.000010385905052645878",
            "extra": "mean: 366.2841622384073 usec\nrounds: 4968"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1590.803655643437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000174362821081222",
            "extra": "mean: 628.6130890210503 usec\nrounds: 1348"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13849.784329004158,
            "unit": "iter/sec",
            "range": "stddev: 0.000012758795234543406",
            "extra": "mean: 72.20329040834262 usec\nrounds: 10895"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1333.2006197877379,
            "unit": "iter/sec",
            "range": "stddev: 0.0001682951725735718",
            "extra": "mean: 750.074658800573 usec\nrounds: 1017"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1348.0449539530382,
            "unit": "iter/sec",
            "range": "stddev: 0.000025380895383010575",
            "extra": "mean: 741.81502409662 usec\nrounds: 1328"
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
          "id": "327736d5de115a09c94f8c2fca0cbd6037648c99",
          "message": "feat: add code_tree.repo_tree() for GitHub repo graph building (0.6.13)\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T19:12:56+02:00",
          "tree_id": "4da33ba1edca9a36aa44b520bb8375efceded797",
          "url": "https://github.com/kkollsga/kglite/commit/327736d5de115a09c94f8c2fca0cbd6037648c99"
        },
        "date": 1774890906972,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1090.2925468622984,
            "unit": "iter/sec",
            "range": "stddev: 0.00001738079288594266",
            "extra": "mean: 917.1850278880222 usec\nrounds: 502"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 800.3699200572024,
            "unit": "iter/sec",
            "range": "stddev: 0.00002409398261168422",
            "extra": "mean: 1.2494222670543764 msec\nrounds: 689"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13928.615504349273,
            "unit": "iter/sec",
            "range": "stddev: 0.000004242303639444463",
            "extra": "mean: 71.79464460683444 usec\nrounds: 7389"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1604.4598786487002,
            "unit": "iter/sec",
            "range": "stddev: 0.00002045324951697199",
            "extra": "mean: 623.262702487902 usec\nrounds: 884"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 706957.6338181832,
            "unit": "iter/sec",
            "range": "stddev: 4.0824368769773904e-7",
            "extra": "mean: 1.4145119200412823 usec\nrounds: 118540"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 129617.6744773846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011820126090180926",
            "extra": "mean: 7.714997233455826 usec\nrounds: 25302"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2787.9958659293616,
            "unit": "iter/sec",
            "range": "stddev: 0.000012588485438665836",
            "extra": "mean: 358.6805892435052 usec\nrounds: 4667"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1553.2448419096472,
            "unit": "iter/sec",
            "range": "stddev: 0.00010111161175862019",
            "extra": "mean: 643.8135012703749 usec\nrounds: 1181"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14862.947124005403,
            "unit": "iter/sec",
            "range": "stddev: 0.000004111698222856311",
            "extra": "mean: 67.28140735863097 usec\nrounds: 12149"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1311.3218015239142,
            "unit": "iter/sec",
            "range": "stddev: 0.00038048868496930687",
            "extra": "mean: 762.5893192943785 usec\nrounds: 1021"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1374.742395957066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158112724934591",
            "extra": "mean: 727.409006182443 usec\nrounds: 1294"
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
          "id": "c8c953f7fdb70fa30e516601ac46195588354ea5",
          "message": "feat: add token param to repo_tree for private repos\n\nSupports token= argument and GITHUB_TOKEN env var fallback.\nToken is scrubbed from verbose output and error messages.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T19:17:24+02:00",
          "tree_id": "a40cd319a9b839d09736b7a2af07b1392f2102ea",
          "url": "https://github.com/kkollsga/kglite/commit/c8c953f7fdb70fa30e516601ac46195588354ea5"
        },
        "date": 1774891174999,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1066.8659231183872,
            "unit": "iter/sec",
            "range": "stddev: 0.000019790338366206753",
            "extra": "mean: 937.3249049674938 usec\nrounds: 463"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 789.5741542269047,
            "unit": "iter/sec",
            "range": "stddev: 0.0001524215014873078",
            "extra": "mean: 1.2665054886188991 msec\nrounds: 659"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 14304.738947962434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000045334400478231625",
            "extra": "mean: 69.90690313453361 usec\nrounds: 6060"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1644.69849454704,
            "unit": "iter/sec",
            "range": "stddev: 0.00002451583956764522",
            "extra": "mean: 608.0141760422819 usec\nrounds: 960"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 691745.3562827448,
            "unit": "iter/sec",
            "range": "stddev: 4.23737549982491e-7",
            "extra": "mean: 1.4456186672126483 usec\nrounds: 113418"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 135252.22809839682,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122214546224174",
            "extra": "mean: 7.3935935404516515 usec\nrounds: 21301"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2896.060758117445,
            "unit": "iter/sec",
            "range": "stddev: 0.000009214501446453584",
            "extra": "mean: 345.29662307569816 usec\nrounds: 5001"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1588.755749969516,
            "unit": "iter/sec",
            "range": "stddev: 0.00003639172141251912",
            "extra": "mean: 629.4233710997977 usec\nrounds: 1218"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14744.455461374473,
            "unit": "iter/sec",
            "range": "stddev: 0.00000426871441381083",
            "extra": "mean: 67.82210456124776 usec\nrounds: 12146"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1333.6455269128776,
            "unit": "iter/sec",
            "range": "stddev: 0.00007293535962791154",
            "extra": "mean: 749.8244322198567 usec\nrounds: 1018"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1324.08903613519,
            "unit": "iter/sec",
            "range": "stddev: 0.00001677111512169054",
            "extra": "mean: 755.2362210617229 usec\nrounds: 1244"
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
          "id": "0c25547b95cf81e4c6f5ecf8c4975dd1925793d5",
          "message": "chore: bump version to 0.6.14\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T19:18:22+02:00",
          "tree_id": "f02ec507d3a693db2f45a4275c5d064a1e571c16",
          "url": "https://github.com/kkollsga/kglite/commit/0c25547b95cf81e4c6f5ecf8c4975dd1925793d5"
        },
        "date": 1774891232231,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1157.8534727465199,
            "unit": "iter/sec",
            "range": "stddev: 0.000023441999479983797",
            "extra": "mean: 863.6671422921253 usec\nrounds: 506"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 903.9898724952739,
            "unit": "iter/sec",
            "range": "stddev: 0.00002265316844331026",
            "extra": "mean: 1.1062070830945376 msec\nrounds: 698"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 14278.181811944964,
            "unit": "iter/sec",
            "range": "stddev: 0.000002617316562226083",
            "extra": "mean: 70.0369285929257 usec\nrounds: 6680"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1568.712737502933,
            "unit": "iter/sec",
            "range": "stddev: 0.00002040396712871574",
            "extra": "mean: 637.4653409086188 usec\nrounds: 924"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 642655.1672799299,
            "unit": "iter/sec",
            "range": "stddev: 2.687573933471931e-7",
            "extra": "mean: 1.5560444401817386 usec\nrounds: 114851"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 134265.92517550194,
            "unit": "iter/sec",
            "range": "stddev: 7.444701092994882e-7",
            "extra": "mean: 7.447906076637674 usec\nrounds: 20719"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2896.0902721430657,
            "unit": "iter/sec",
            "range": "stddev: 0.000007751560005424424",
            "extra": "mean: 345.29310416143005 usec\nrounds: 3965"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1559.5612341381266,
            "unit": "iter/sec",
            "range": "stddev: 0.000032969290461607276",
            "extra": "mean: 641.2059867291061 usec\nrounds: 1281"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14339.18518884262,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028932598900884365",
            "extra": "mean: 69.73896960185047 usec\nrounds: 9573"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1498.0798068666325,
            "unit": "iter/sec",
            "range": "stddev: 0.00014165789694726757",
            "extra": "mean: 667.5211797237887 usec\nrounds: 1085"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1490.5186493698081,
            "unit": "iter/sec",
            "range": "stddev: 0.00006987680759111222",
            "extra": "mean: 670.9074055683908 usec\nrounds: 1329"
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
          "id": "fb144573023026d442d8a5cf9793bf13173754e6",
          "message": "fix: ruff format repo.py\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T19:35:46+02:00",
          "tree_id": "48eae05b2f5ae17ec10218768fa7c9e16e6ee949",
          "url": "https://github.com/kkollsga/kglite/commit/fb144573023026d442d8a5cf9793bf13173754e6"
        },
        "date": 1774892274206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1067.9431464057134,
            "unit": "iter/sec",
            "range": "stddev: 0.000023543751644590466",
            "extra": "mean: 936.3794349592636 usec\nrounds: 492"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 788.6510350437901,
            "unit": "iter/sec",
            "range": "stddev: 0.000026683291036778195",
            "extra": "mean: 1.267987938346489 msec\nrounds: 665"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13312.367831797446,
            "unit": "iter/sec",
            "range": "stddev: 0.000018671635975887318",
            "extra": "mean: 75.11811667428807 usec\nrounds: 6531"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1587.2460044803179,
            "unit": "iter/sec",
            "range": "stddev: 0.000023075248261554132",
            "extra": "mean: 630.0220615942966 usec\nrounds: 828"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 690069.1236576899,
            "unit": "iter/sec",
            "range": "stddev: 4.664176970561694e-7",
            "extra": "mean: 1.4491301895954005 usec\nrounds: 114195"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 136892.30318824196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011676544108790598",
            "extra": "mean: 7.305012602680007 usec\nrounds: 20551"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2865.148650571929,
            "unit": "iter/sec",
            "range": "stddev: 0.000010736148677239775",
            "extra": "mean: 349.02203060228106 usec\nrounds: 5065"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1555.4429768325388,
            "unit": "iter/sec",
            "range": "stddev: 0.000025175504686682916",
            "extra": "mean: 642.9036711049173 usec\nrounds: 1277"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14201.281027469304,
            "unit": "iter/sec",
            "range": "stddev: 0.000004859609315467151",
            "extra": "mean: 70.41618274194535 usec\nrounds: 11809"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1343.886450374423,
            "unit": "iter/sec",
            "range": "stddev: 0.00006310516566851477",
            "extra": "mean: 744.1104862106378 usec\nrounds: 979"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1362.0495619350738,
            "unit": "iter/sec",
            "range": "stddev: 0.00001544945088304526",
            "extra": "mean: 734.1876741836712 usec\nrounds: 1286"
          }
        ]
      }
    ]
  }
}