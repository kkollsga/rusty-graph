window.BENCHMARK_DATA = {
  "lastUpdate": 1776630446996,
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
          "id": "e9e75d539702d329f001f69f4878c3d6534f66c6",
          "message": "fix: ruff import sorting in repo.py\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T19:40:05+02:00",
          "tree_id": "4099f18d8911c0caf569d3f513444919fc82267a",
          "url": "https://github.com/kkollsga/kglite/commit/e9e75d539702d329f001f69f4878c3d6534f66c6"
        },
        "date": 1774892558205,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1102.2668548442368,
            "unit": "iter/sec",
            "range": "stddev: 0.000017917441578526824",
            "extra": "mean: 907.2213281250407 usec\nrounds: 512"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 804.3684168502238,
            "unit": "iter/sec",
            "range": "stddev: 0.000030445839913845645",
            "extra": "mean: 1.2432114178672478 msec\nrounds: 694"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13863.463987907298,
            "unit": "iter/sec",
            "range": "stddev: 0.000004366269007928521",
            "extra": "mean: 72.13204440623724 usec\nrounds: 6981"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1628.5433592107588,
            "unit": "iter/sec",
            "range": "stddev: 0.000020294721974738985",
            "extra": "mean: 614.045671147884 usec\nrounds: 967"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 702764.8756140787,
            "unit": "iter/sec",
            "range": "stddev: 4.4549438572034363e-7",
            "extra": "mean: 1.4229510248732853 usec\nrounds: 130311"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 136818.64440419406,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012896023651909588",
            "extra": "mean: 7.30894538792365 usec\nrounds: 21314"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2785.3811907679824,
            "unit": "iter/sec",
            "range": "stddev: 0.00001404607249940367",
            "extra": "mean: 359.0172875850723 usec\nrounds: 4277"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1579.4499116049105,
            "unit": "iter/sec",
            "range": "stddev: 0.000025575388073633316",
            "extra": "mean: 633.1318218150268 usec\nrounds: 1201"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14044.208677968905,
            "unit": "iter/sec",
            "range": "stddev: 0.000006003026705367213",
            "extra": "mean: 71.20372695463405 usec\nrounds: 9427"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1293.0753394652368,
            "unit": "iter/sec",
            "range": "stddev: 0.0002977842916607655",
            "extra": "mean: 773.3501440167896 usec\nrounds: 986"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1327.9875015802497,
            "unit": "iter/sec",
            "range": "stddev: 0.000016254302318722882",
            "extra": "mean: 753.0191352027348 usec\nrounds: 1213"
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
          "id": "3fb47aa3f5549b6632100d9c279337481be4db87",
          "message": "fix: auto-create stub nodes for external base classes in EXTENDS edges\n\nSame pattern as external traits — when a class extends a base class\nfrom an external library, create a stub Class node so the edge\nconnects properly instead of being silently skipped.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T20:13:12+02:00",
          "tree_id": "e68da9a258c8cea7b1111a186f1ec20cd66716b4",
          "url": "https://github.com/kkollsga/kglite/commit/3fb47aa3f5549b6632100d9c279337481be4db87"
        },
        "date": 1774894521051,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1177.7703777559457,
            "unit": "iter/sec",
            "range": "stddev: 0.000021362794391761854",
            "extra": "mean: 849.0619384615032 usec\nrounds: 455"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 852.4647856430726,
            "unit": "iter/sec",
            "range": "stddev: 0.00002970920571687049",
            "extra": "mean: 1.1730689840116169 msec\nrounds: 688"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13345.554985481414,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032716663536568454",
            "extra": "mean: 74.93131616391351 usec\nrounds: 5339"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1503.3748921014696,
            "unit": "iter/sec",
            "range": "stddev: 0.00003083709477767803",
            "extra": "mean: 665.1700818297992 usec\nrounds: 831"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 659151.1084865116,
            "unit": "iter/sec",
            "range": "stddev: 4.2167088914178954e-7",
            "extra": "mean: 1.517102811669569 usec\nrounds: 105776"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 129337.06826616483,
            "unit": "iter/sec",
            "range": "stddev: 9.346872683774458e-7",
            "extra": "mean: 7.731735483149222 usec\nrounds: 17583"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2652.908114463841,
            "unit": "iter/sec",
            "range": "stddev: 0.000008528699213349353",
            "extra": "mean: 376.9448306739047 usec\nrounds: 3886"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1476.5484910679795,
            "unit": "iter/sec",
            "range": "stddev: 0.000026095470180842533",
            "extra": "mean: 677.255102727243 usec\nrounds: 1100"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13917.166796936866,
            "unit": "iter/sec",
            "range": "stddev: 0.000015026693683695524",
            "extra": "mean: 71.85370518230029 usec\nrounds: 10169"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1200.4721793879105,
            "unit": "iter/sec",
            "range": "stddev: 0.000018786072744400692",
            "extra": "mean: 833.0055599538125 usec\nrounds: 859"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1221.8113021424058,
            "unit": "iter/sec",
            "range": "stddev: 0.000014627176496021253",
            "extra": "mean: 818.456989427527 usec\nrounds: 1135"
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
          "id": "617668216b6c70d2b4ba11442d16e30f351c66a9",
          "message": "feat: kglite.repo_tree() + fix code_tree edge skip warnings (0.6.15)\n\n- Re-export repo_tree at top-level: kglite.repo_tree(\"org/repo\")\n- Auto-create stub nodes for external base classes, enums, and traits\n  in EXTENDS/IMPLEMENTS/HAS_METHOD edges\n- Include enums in type routing maps and name resolution\n- Register external traits in name_to_qname for owner resolution\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T20:20:13+02:00",
          "tree_id": "ff68b9d915c754d6269f276525b0769eb1d41a4d",
          "url": "https://github.com/kkollsga/kglite/commit/617668216b6c70d2b4ba11442d16e30f351c66a9"
        },
        "date": 1774894946344,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1002.0431788887435,
            "unit": "iter/sec",
            "range": "stddev: 0.0002302608479807765",
            "extra": "mean: 997.960987179206 usec\nrounds: 468"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 798.7569603222541,
            "unit": "iter/sec",
            "range": "stddev: 0.000033173659401894804",
            "extra": "mean: 1.2519452720594202 msec\nrounds: 680"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13524.389858809769,
            "unit": "iter/sec",
            "range": "stddev: 0.0000057436265184674065",
            "extra": "mean: 73.94048903053482 usec\nrounds: 7065"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1567.3540357597187,
            "unit": "iter/sec",
            "range": "stddev: 0.00013083989628510776",
            "extra": "mean: 638.0179443729098 usec\nrounds: 773"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 708483.8422024025,
            "unit": "iter/sec",
            "range": "stddev: 4.71085073892595e-7",
            "extra": "mean: 1.411464793454409 usec\nrounds: 113033"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 132414.46398543313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013325200890313447",
            "extra": "mean: 7.552045070469111 usec\nrounds: 20834"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2873.7208800316293,
            "unit": "iter/sec",
            "range": "stddev: 0.000040344218244943986",
            "extra": "mean: 347.9809075921784 usec\nrounds: 4913"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1577.817720936519,
            "unit": "iter/sec",
            "range": "stddev: 0.00004744191416303301",
            "extra": "mean: 633.786771900652 usec\nrounds: 1210"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14516.57072291324,
            "unit": "iter/sec",
            "range": "stddev: 0.000004572159117219362",
            "extra": "mean: 68.88679283059466 usec\nrounds: 11884"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1346.0794562702988,
            "unit": "iter/sec",
            "range": "stddev: 0.00005557386090739877",
            "extra": "mean: 742.8981961962248 usec\nrounds: 999"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1350.1297414278129,
            "unit": "iter/sec",
            "range": "stddev: 0.000015675168242886973",
            "extra": "mean: 740.6695588695517 usec\nrounds: 1274"
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
          "id": "085bef9a702a4e90d589fe5d9be150ff473c4e48",
          "message": "fix: gate code_tree parse output on verbose flag\n\nThe \"Found N files\" message printed unconditionally during parsing.\nNow respects verbose=False for silent operation.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T20:46:36+02:00",
          "tree_id": "5b0beb801d6f81d1a01c149dadc686c6a8f98d38",
          "url": "https://github.com/kkollsga/kglite/commit/085bef9a702a4e90d589fe5d9be150ff473c4e48"
        },
        "date": 1774896527461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1043.7302480702335,
            "unit": "iter/sec",
            "range": "stddev: 0.000017267075396675678",
            "extra": "mean: 958.1019634612613 usec\nrounds: 520"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 798.1781259937568,
            "unit": "iter/sec",
            "range": "stddev: 0.000027345518666022596",
            "extra": "mean: 1.2528531757932713 msec\nrounds: 694"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13838.980130346863,
            "unit": "iter/sec",
            "range": "stddev: 0.000004252977066170525",
            "extra": "mean: 72.25966007474395 usec\nrounds: 7484"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1541.4316323819494,
            "unit": "iter/sec",
            "range": "stddev: 0.000024129405966496763",
            "extra": "mean: 648.7475532435494 usec\nrounds: 817"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 703312.0373497012,
            "unit": "iter/sec",
            "range": "stddev: 4.2699095977918013e-7",
            "extra": "mean: 1.421843999383704 usec\nrounds: 126679"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130630.02323765085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012475192488110388",
            "extra": "mean: 7.65520800819834 usec\nrounds: 30619"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2709.7770008539255,
            "unit": "iter/sec",
            "range": "stddev: 0.000012025786751771944",
            "extra": "mean: 369.0340569297298 usec\nrounds: 5094"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1554.5086025453293,
            "unit": "iter/sec",
            "range": "stddev: 0.00002464909508915204",
            "extra": "mean: 643.2901036138461 usec\nrounds: 1245"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13729.651537572812,
            "unit": "iter/sec",
            "range": "stddev: 0.000013540859331805396",
            "extra": "mean: 72.8350604721017 usec\nrounds: 10418"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1333.2230365975738,
            "unit": "iter/sec",
            "range": "stddev: 0.00017848004384651034",
            "extra": "mean: 750.0620470465548 usec\nrounds: 999"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1342.1243698699416,
            "unit": "iter/sec",
            "range": "stddev: 0.00001621730029986069",
            "extra": "mean: 745.0874318725804 usec\nrounds: 1255"
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
          "id": "d4b4ab946e5844e5244acb4ee76aa19ef88df3f8",
          "message": "feat: Polars-style ResultView display + improved docs (0.6.16)\n\n- ResultView repr/str now shows a bordered table with shape header\n- Large results show first 10 + last 5 rows with … separator\n- help(ResultView) includes quick-reference cheat sheet\n- code_tree parse output respects verbose flag\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T21:15:39+02:00",
          "tree_id": "20645c4ae8979b6e7cb3d600cbbd73fa7198ac39",
          "url": "https://github.com/kkollsga/kglite/commit/d4b4ab946e5844e5244acb4ee76aa19ef88df3f8"
        },
        "date": 1774898267001,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1147.609353345743,
            "unit": "iter/sec",
            "range": "stddev: 0.0001722096048610969",
            "extra": "mean: 871.3766553789384 usec\nrounds: 502"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 911.800834144751,
            "unit": "iter/sec",
            "range": "stddev: 0.000021997357212626076",
            "extra": "mean: 1.0967307360910434 msec\nrounds: 701"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 15179.116147741544,
            "unit": "iter/sec",
            "range": "stddev: 0.000001947937168713659",
            "extra": "mean: 65.87998868094748 usec\nrounds: 6891"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1554.8011736284666,
            "unit": "iter/sec",
            "range": "stddev: 0.000021096603504653514",
            "extra": "mean: 643.1690539995429 usec\nrounds: 1000"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 673349.7801812965,
            "unit": "iter/sec",
            "range": "stddev: 2.8718218696428914e-7",
            "extra": "mean: 1.4851122394823597 usec\nrounds: 123290"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 133394.7617676546,
            "unit": "iter/sec",
            "range": "stddev: 6.464318137063572e-7",
            "extra": "mean: 7.496546241761636 usec\nrounds: 22404"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2981.655372604723,
            "unit": "iter/sec",
            "range": "stddev: 0.000007547622133158228",
            "extra": "mean: 335.38416585227856 usec\nrounds: 3889"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1536.3340899407208,
            "unit": "iter/sec",
            "range": "stddev: 0.000055351962682930936",
            "extra": "mean: 650.900091684215 usec\nrounds: 938"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 15290.321280181734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028163123964818767",
            "extra": "mean: 65.40084944429071 usec\nrounds: 10707"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1487.4850978519348,
            "unit": "iter/sec",
            "range": "stddev: 0.000014507941883614024",
            "extra": "mean: 672.2756425890195 usec\nrounds: 1066"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1515.352316957814,
            "unit": "iter/sec",
            "range": "stddev: 0.000010921299523886817",
            "extra": "mean: 659.9125423238714 usec\nrounds: 1394"
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
          "id": "c8719a8125cb5c35b6e86582ae1a3df0ffbdf0f0",
          "message": "feat: to_neo4j(), Polars-style ResultView, verbose fix (0.6.17)\n\n- kglite.to_neo4j(graph, uri) — direct Neo4j push via batched UNWIND\n- ResultView repr shows bordered table with shape header\n- help(ResultView) includes quick-reference cheat sheet\n- code_tree parse output respects verbose flag\n- neo4j optional dependency in pyproject.toml\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T23:02:02+02:00",
          "tree_id": "b8bc0db07cca08cb90c99c2077e4732aa7a7c264",
          "url": "https://github.com/kkollsga/kglite/commit/c8719a8125cb5c35b6e86582ae1a3df0ffbdf0f0"
        },
        "date": 1774904654147,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1161.9237710438676,
            "unit": "iter/sec",
            "range": "stddev: 0.000044305770188142053",
            "extra": "mean: 860.6416573280054 usec\nrounds: 464"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 835.6938398209488,
            "unit": "iter/sec",
            "range": "stddev: 0.000030160984134542714",
            "extra": "mean: 1.1966104718616262 msec\nrounds: 693"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12617.558962529667,
            "unit": "iter/sec",
            "range": "stddev: 0.000013023198482005944",
            "extra": "mean: 79.25463260918356 usec\nrounds: 6078"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1514.1491715998668,
            "unit": "iter/sec",
            "range": "stddev: 0.00002766674784853203",
            "extra": "mean: 660.4369098874115 usec\nrounds: 799"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 659733.0678698977,
            "unit": "iter/sec",
            "range": "stddev: 3.987532505235342e-7",
            "extra": "mean: 1.515764554941491 usec\nrounds: 111062"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 136101.81604835283,
            "unit": "iter/sec",
            "range": "stddev: 9.760986302443736e-7",
            "extra": "mean: 7.347440534112567 usec\nrounds: 18271"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2703.7583898457247,
            "unit": "iter/sec",
            "range": "stddev: 0.000020824712475263316",
            "extra": "mean: 369.85553285959827 usec\nrounds: 4504"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1466.155342861985,
            "unit": "iter/sec",
            "range": "stddev: 0.00007406300882778441",
            "extra": "mean: 682.0559668990913 usec\nrounds: 1148"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13503.561612869391,
            "unit": "iter/sec",
            "range": "stddev: 0.00001956174697427972",
            "extra": "mean: 74.0545367710222 usec\nrounds: 10239"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1216.8538755188315,
            "unit": "iter/sec",
            "range": "stddev: 0.000022548087785957928",
            "extra": "mean: 821.7913589449092 usec\nrounds: 872"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1203.604041522049,
            "unit": "iter/sec",
            "range": "stddev: 0.000015308946400441396",
            "extra": "mean: 830.8380210616638 usec\nrounds: 1092"
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
          "id": "0878fe4c1d55bfa7d63f4bb644d12207a8685a10",
          "message": "fix: 16x faster multi-hop LIMIT via pattern matcher pushdown (0.6.18)\n\n- Planner: push LIMIT into MATCH for edge patterns (was node-only)\n- Executor: post-match row truncation before Return projection\n- Pattern matcher: last-hop early termination + intermediate overcommit\n- Added window function safety check to LIMIT fusion guard\n\nBenchmarks (2-hop LIMIT 20): 16,538μs → 1,008μs (16x speedup)\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T23:23:03+02:00",
          "tree_id": "de7593e0913d32dfc780e56b00ab73ef5cd3ffe9",
          "url": "https://github.com/kkollsga/kglite/commit/0878fe4c1d55bfa7d63f4bb644d12207a8685a10"
        },
        "date": 1774905911077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1064.4237781900197,
            "unit": "iter/sec",
            "range": "stddev: 0.00003269804322286199",
            "extra": "mean: 939.4754424787767 usec\nrounds: 452"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 793.6301331469443,
            "unit": "iter/sec",
            "range": "stddev: 0.000029798043661746004",
            "extra": "mean: 1.2600328014698066 msec\nrounds: 680"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13721.891589307643,
            "unit": "iter/sec",
            "range": "stddev: 0.000004936578035801336",
            "extra": "mean: 72.87624985896397 usec\nrounds: 7100"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1568.446780555645,
            "unit": "iter/sec",
            "range": "stddev: 0.000022708795419567574",
            "extra": "mean: 637.5734340477497 usec\nrounds: 887"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 683390.2672806695,
            "unit": "iter/sec",
            "range": "stddev: 0.000003330053022149996",
            "extra": "mean: 1.4632927155652604 usec\nrounds: 115795"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 133380.91240339272,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122946601674673",
            "extra": "mean: 7.497324631995572 usec\nrounds: 22142"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2898.0827164103266,
            "unit": "iter/sec",
            "range": "stddev: 0.000011241312453995104",
            "extra": "mean: 345.055713674949 usec\nrounds: 4680"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1585.3669776759043,
            "unit": "iter/sec",
            "range": "stddev: 0.00002210899011623874",
            "extra": "mean: 630.7687835569573 usec\nrounds: 1192"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14769.891216865066,
            "unit": "iter/sec",
            "range": "stddev: 0.000004684405262368587",
            "extra": "mean: 67.7053057004337 usec\nrounds: 11894"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1285.216975548991,
            "unit": "iter/sec",
            "range": "stddev: 0.0003403565875763628",
            "extra": "mean: 778.0787361393525 usec\nrounds: 974"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1324.9361274112125,
            "unit": "iter/sec",
            "range": "stddev: 0.00002522273985705722",
            "extra": "mean: 754.753364567012 usec\nrounds: 1270"
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
          "id": "0a8b0376b054a87b275c27ff56db8d0a51f5940f",
          "message": "fix",
          "timestamp": "2026-03-30T23:26:20+02:00",
          "tree_id": "52171cf6ef2e033c324a901d3add4cdd5fbbd84b",
          "url": "https://github.com/kkollsga/kglite/commit/0a8b0376b054a87b275c27ff56db8d0a51f5940f"
        },
        "date": 1774906110676,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1066.5230891132317,
            "unit": "iter/sec",
            "range": "stddev: 0.000015812362824633158",
            "extra": "mean: 937.6262081971964 usec\nrounds: 610"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 795.3749047432065,
            "unit": "iter/sec",
            "range": "stddev: 0.00002073343753098255",
            "extra": "mean: 1.257268734576003 msec\nrounds: 697"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13351.521683742607,
            "unit": "iter/sec",
            "range": "stddev: 0.000004387794111601996",
            "extra": "mean: 74.8978299018638 usec\nrounds: 7237"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1628.6402187713354,
            "unit": "iter/sec",
            "range": "stddev: 0.00009634258886523357",
            "extra": "mean: 614.0091522205017 usec\nrounds: 946"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 698779.9544156453,
            "unit": "iter/sec",
            "range": "stddev: 4.5926526007261944e-7",
            "extra": "mean: 1.4310656647789073 usec\nrounds: 130481"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 133287.0885676187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012076360885515308",
            "extra": "mean: 7.50260217059722 usec\nrounds: 30684"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2854.2404423505964,
            "unit": "iter/sec",
            "range": "stddev: 0.000027269179046389195",
            "extra": "mean: 350.355907358826 usec\nrounds: 4987"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1643.0069058178165,
            "unit": "iter/sec",
            "range": "stddev: 0.00002193804659810822",
            "extra": "mean: 608.6401684977972 usec\nrounds: 1365"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14624.013240882632,
            "unit": "iter/sec",
            "range": "stddev: 0.000007391512310500661",
            "extra": "mean: 68.38068206916128 usec\nrounds: 11946"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1284.7329230269972,
            "unit": "iter/sec",
            "range": "stddev: 0.00042580493463742014",
            "extra": "mean: 778.3718951047588 usec\nrounds: 1001"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1345.0149541973512,
            "unit": "iter/sec",
            "range": "stddev: 0.000022553882968350835",
            "extra": "mean: 743.4861574433261 usec\nrounds: 1283"
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
          "id": "caf59b00a55f2b4a6e7af5d001a844c02e9e14a9",
          "message": "fix",
          "timestamp": "2026-03-30T23:31:28+02:00",
          "tree_id": "8c51f31e280e83a6521e6ebd8a722170f42a5d5b",
          "url": "https://github.com/kkollsga/kglite/commit/caf59b00a55f2b4a6e7af5d001a844c02e9e14a9"
        },
        "date": 1774906414323,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1095.9214217517133,
            "unit": "iter/sec",
            "range": "stddev: 0.00001693957630111728",
            "extra": "mean: 912.4741794002044 usec\nrounds: 602"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 798.868348516099,
            "unit": "iter/sec",
            "range": "stddev: 0.000036356222521645765",
            "extra": "mean: 1.2517707102271654 msec\nrounds: 704"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13962.076521594394,
            "unit": "iter/sec",
            "range": "stddev: 0.000003927076872283537",
            "extra": "mean: 71.6225841087 usec\nrounds: 6557"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1629.5399050451922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000171911724597246",
            "extra": "mean: 613.6701512518448 usec\nrounds: 1038"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 696723.4186273356,
            "unit": "iter/sec",
            "range": "stddev: 4.453736531304359e-7",
            "extra": "mean: 1.4352897767813966 usec\nrounds: 128140"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130947.148364035,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012737723572381634",
            "extra": "mean: 7.636668781972901 usec\nrounds: 33096"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2774.636866794039,
            "unit": "iter/sec",
            "range": "stddev: 0.00001189828154685175",
            "extra": "mean: 360.40752286098336 usec\nrounds: 5074"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1500.547317373186,
            "unit": "iter/sec",
            "range": "stddev: 0.00011850201513425616",
            "extra": "mean: 666.4235032258567 usec\nrounds: 1085"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14078.369741827015,
            "unit": "iter/sec",
            "range": "stddev: 0.00000399000542958598",
            "extra": "mean: 71.03095161856612 usec\nrounds: 11182"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1345.8482334595794,
            "unit": "iter/sec",
            "range": "stddev: 0.00002034507757383287",
            "extra": "mean: 743.0258294647705 usec\nrounds: 991"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1333.1873493190294,
            "unit": "iter/sec",
            "range": "stddev: 0.00001837207861612269",
            "extra": "mean: 750.0821249997488 usec\nrounds: 1272"
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
          "id": "aa969cb4c01d788676d53853ca50a80f0182a2be",
          "message": "fix",
          "timestamp": "2026-03-30T23:45:13+02:00",
          "tree_id": "fe432e7d81c1f3fbcfa281c4d2caa06a2e348211",
          "url": "https://github.com/kkollsga/kglite/commit/aa969cb4c01d788676d53853ca50a80f0182a2be"
        },
        "date": 1774907240589,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1003.5237025164699,
            "unit": "iter/sec",
            "range": "stddev: 0.0001577593823165616",
            "extra": "mean: 996.4886703646024 usec\nrounds: 631"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 789.759432514178,
            "unit": "iter/sec",
            "range": "stddev: 0.000024384787783545333",
            "extra": "mean: 1.2662083652695695 msec\nrounds: 668"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13698.276895837804,
            "unit": "iter/sec",
            "range": "stddev: 0.000004252429321195743",
            "extra": "mean: 73.00188247062286 usec\nrounds: 6492"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1581.1570160463707,
            "unit": "iter/sec",
            "range": "stddev: 0.000021050965177610232",
            "extra": "mean: 632.4482577324711 usec\nrounds: 970"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 700232.2127404744,
            "unit": "iter/sec",
            "range": "stddev: 4.210561204904714e-7",
            "extra": "mean: 1.4280976821765097 usec\nrounds: 125550"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131023.11586198585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011473924796964592",
            "extra": "mean: 7.632241024197267 usec\nrounds: 33117"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2753.2046624194113,
            "unit": "iter/sec",
            "range": "stddev: 0.000010476338518151488",
            "extra": "mean: 363.21309986495453 usec\nrounds: 4436"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1579.374780300108,
            "unit": "iter/sec",
            "range": "stddev: 0.000024259527146045022",
            "extra": "mean: 633.1619400747827 usec\nrounds: 1335"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14250.915772448552,
            "unit": "iter/sec",
            "range": "stddev: 0.000016269430637663607",
            "extra": "mean: 70.17092908045325 usec\nrounds: 11224"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1316.9075151080037,
            "unit": "iter/sec",
            "range": "stddev: 0.00010831721424350152",
            "extra": "mean: 759.3547675350511 usec\nrounds: 998"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1314.0282934871652,
            "unit": "iter/sec",
            "range": "stddev: 0.000014622588195653962",
            "extra": "mean: 761.0186211030527 usec\nrounds: 1251"
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
          "id": "f57949088e7b26f9a605db7a3503f4e8ef42e2e2",
          "message": "stubtest fix",
          "timestamp": "2026-04-03T09:43:12+02:00",
          "tree_id": "fad2acdd9a2ccaebf78159a5db857c76406fde0a",
          "url": "https://github.com/kkollsga/kglite/commit/f57949088e7b26f9a605db7a3503f4e8ef42e2e2"
        },
        "date": 1775202327855,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1063.170484178392,
            "unit": "iter/sec",
            "range": "stddev: 0.000023735337262801134",
            "extra": "mean: 940.5829214425479 usec\nrounds: 471"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 768.3658169846979,
            "unit": "iter/sec",
            "range": "stddev: 0.000032906547999751927",
            "extra": "mean: 1.301463414814971 msec\nrounds: 675"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13649.350775731893,
            "unit": "iter/sec",
            "range": "stddev: 0.000004670198175367998",
            "extra": "mean: 73.26355783734182 usec\nrounds: 7158"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1584.6807906598056,
            "unit": "iter/sec",
            "range": "stddev: 0.000026163354803380845",
            "extra": "mean: 631.0419144941078 usec\nrounds: 959"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 655684.2065626314,
            "unit": "iter/sec",
            "range": "stddev: 5.986554575087367e-7",
            "extra": "mean: 1.5251244272641777 usec\nrounds: 123534"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 135445.57360912443,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011183030127323526",
            "extra": "mean: 7.383039351923377 usec\nrounds: 21727"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2782.4654407630856,
            "unit": "iter/sec",
            "range": "stddev: 0.000012490295203137441",
            "extra": "mean: 359.3935023774283 usec\nrounds: 4837"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1547.182562533922,
            "unit": "iter/sec",
            "range": "stddev: 0.00002080148116961153",
            "extra": "mean: 646.3361365463132 usec\nrounds: 1245"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13670.718932124124,
            "unit": "iter/sec",
            "range": "stddev: 0.000016727260638324953",
            "extra": "mean: 73.14904248745478 usec\nrounds: 10662"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1283.7921021571335,
            "unit": "iter/sec",
            "range": "stddev: 0.0002230002262473446",
            "extra": "mean: 778.942321205838 usec\nrounds: 962"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1309.9902570115607,
            "unit": "iter/sec",
            "range": "stddev: 0.00001773288012047198",
            "extra": "mean: 763.3644560695194 usec\nrounds: 1252"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58362424+kkollsga@users.noreply.github.com",
            "name": "kkollsga",
            "username": "kkollsga"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9df6f85abc78632ace9ac7be2f4e2cb080ecec77",
          "message": "Merge pull request #8 from kkollsga/dependabot/github_actions/github-actions-fc637c5bdb\n\nchore: Bump the github-actions group with 2 updates",
          "timestamp": "2026-04-03T09:56:06+02:00",
          "tree_id": "c3777d839af257c1bb26fb6c1a5ef77ff7f19ba4",
          "url": "https://github.com/kkollsga/kglite/commit/9df6f85abc78632ace9ac7be2f4e2cb080ecec77"
        },
        "date": 1775203085508,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1049.5915007944916,
            "unit": "iter/sec",
            "range": "stddev: 0.000026285741811462473",
            "extra": "mean: 952.7516174083411 usec\nrounds: 494"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 776.6923987583838,
            "unit": "iter/sec",
            "range": "stddev: 0.00003584438273167432",
            "extra": "mean: 1.2875109909644984 msec\nrounds: 664"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13576.089376662574,
            "unit": "iter/sec",
            "range": "stddev: 0.000004672897898696841",
            "extra": "mean: 73.65891401090873 usec\nrounds: 7280"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1602.1115599053962,
            "unit": "iter/sec",
            "range": "stddev: 0.00003331659497828947",
            "extra": "mean: 624.1762590234662 usec\nrounds: 942"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 702671.9802504688,
            "unit": "iter/sec",
            "range": "stddev: 4.5726412598292794e-7",
            "extra": "mean: 1.4231391433077323 usec\nrounds: 119977"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 135009.14660039925,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011659891817205046",
            "extra": "mean: 7.406905570330023 usec\nrounds: 21794"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2747.028409758777,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112947336402281",
            "extra": "mean: 364.0297262480123 usec\nrounds: 4968"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1577.9414079699666,
            "unit": "iter/sec",
            "range": "stddev: 0.000021426595958704845",
            "extra": "mean: 633.7370924859039 usec\nrounds: 1211"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13680.503870969063,
            "unit": "iter/sec",
            "range": "stddev: 0.000004502224878329333",
            "extra": "mean: 73.09672285697505 usec\nrounds: 10767"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1315.6791890177471,
            "unit": "iter/sec",
            "range": "stddev: 0.00003700260853411676",
            "extra": "mean: 760.0637057629335 usec\nrounds: 989"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1316.4923471202978,
            "unit": "iter/sec",
            "range": "stddev: 0.000017282459118074245",
            "extra": "mean: 759.5942370553124 usec\nrounds: 1236"
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
          "id": "fc6fed341b7ffa83537f2e195ce0e8185ec3de59",
          "message": "feat: disk storage mode with CSR edge format (0.7.0)\n\nAdd fully disk-backed graph storage for 100M+ node graphs.\nThree interchangeable storage modes: default (heap), mapped (mmap\ncolumnar), and disk (CSR on disk). All share the same API — Cypher,\nfluent API, and algorithms work identically across modes.\n\nKey additions:\n- GraphBackend enum abstracting petgraph behind 22-method interface\n- DiskGraph with CSR edge arrays, mmap'd node slots, edge arena\n- Iterator wrappers (GraphEdgeRef + 6 enum iterators)\n- zstd N-Triples support, 81x faster entity loading\n- Mapped mode O(n²) Arc clone fix (50-300x faster add_nodes)\n- Auto-CSR build on add_connections for seamless queries\n\nDisk mode benchmarks vs default (100k nodes):\n- WHERE+LIMIT: 3.4x faster\n- Fluent select: 3.7x faster\n- SET updates: 2.5x faster\n- Load: 1.7x faster\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T01:32:07+02:00",
          "tree_id": "5498cec74a857ef44594334e941460c20e6417c3",
          "url": "https://github.com/kkollsga/kglite/commit/fc6fed341b7ffa83537f2e195ce0e8185ec3de59"
        },
        "date": 1775345670790,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1130.1336968395456,
            "unit": "iter/sec",
            "range": "stddev: 0.000026880078678797757",
            "extra": "mean: 884.8510603626203 usec\nrounds: 497"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 823.8873474551166,
            "unit": "iter/sec",
            "range": "stddev: 0.000024255212534836705",
            "extra": "mean: 1.2137581710519927 msec\nrounds: 684"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13882.395623291064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019935592662475828",
            "extra": "mean: 72.03367683328797 usec\nrounds: 6492"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1492.8918133089026,
            "unit": "iter/sec",
            "range": "stddev: 0.00011961868522146103",
            "extra": "mean: 669.8409027935935 usec\nrounds: 895"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 639942.2871434172,
            "unit": "iter/sec",
            "range": "stddev: 2.7707024645399714e-7",
            "extra": "mean: 1.5626409132358063 usec\nrounds: 89438"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131863.9073871563,
            "unit": "iter/sec",
            "range": "stddev: 6.051151022809214e-7",
            "extra": "mean: 7.583576278108994 usec\nrounds: 19560"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2678.116052444114,
            "unit": "iter/sec",
            "range": "stddev: 0.00003557249111397047",
            "extra": "mean: 373.3968134380792 usec\nrounds: 4063"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1501.9589806390516,
            "unit": "iter/sec",
            "range": "stddev: 0.00002798221659424451",
            "extra": "mean: 665.7971441899973 usec\nrounds: 1179"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14496.671742598912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023840524486881944",
            "extra": "mean: 68.9813508752819 usec\nrounds: 9881"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1407.0116390572898,
            "unit": "iter/sec",
            "range": "stddev: 0.000013052327620881685",
            "extra": "mean: 710.7261747102596 usec\nrounds: 1036"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1391.7234176591962,
            "unit": "iter/sec",
            "range": "stddev: 0.000011757007432890435",
            "extra": "mean: 718.5335730586082 usec\nrounds: 1314"
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
          "id": "4d19705517d3f014d661f66d7138e1edf2aa7398",
          "message": "fix: CSR build uses mmap files to avoid OOM on large graphs\n\nbuild_csr_from_pending() was allocating CSR arrays on heap (~45 GB\nfor 862M edges). Now writes directly to mmap'd files in data_dir.\nOS manages paging — only hot pages in RAM.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T06:29:06+02:00",
          "tree_id": "e8b364d691d862230633123db407b12d4acc547b",
          "url": "https://github.com/kkollsga/kglite/commit/4d19705517d3f014d661f66d7138e1edf2aa7398"
        },
        "date": 1775363476601,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1035.5650500696584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000247830415891579",
            "extra": "mean: 965.6563824095203 usec\nrounds: 523"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 722.917592239774,
            "unit": "iter/sec",
            "range": "stddev: 0.000029695832973396865",
            "extra": "mean: 1.3832835315319378 msec\nrounds: 666"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13221.124334718344,
            "unit": "iter/sec",
            "range": "stddev: 0.000015955887116351724",
            "extra": "mean: 75.63653246751677 usec\nrounds: 7084"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1567.017541629651,
            "unit": "iter/sec",
            "range": "stddev: 0.000020465368294505558",
            "extra": "mean: 638.1549494079243 usec\nrounds: 929"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 696107.1761590721,
            "unit": "iter/sec",
            "range": "stddev: 4.0022920271305716e-7",
            "extra": "mean: 1.4365603950784203 usec\nrounds: 78475"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128672.37603406457,
            "unit": "iter/sec",
            "range": "stddev: 0.00000131856538223938",
            "extra": "mean: 7.771675870314707 usec\nrounds: 34443"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2537.287623657894,
            "unit": "iter/sec",
            "range": "stddev: 0.00005635167908968739",
            "extra": "mean: 394.12165600616646 usec\nrounds: 3971"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1553.1816081997763,
            "unit": "iter/sec",
            "range": "stddev: 0.00002215174871907801",
            "extra": "mean: 643.8397124461546 usec\nrounds: 1165"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13887.608239547695,
            "unit": "iter/sec",
            "range": "stddev: 0.000005174032196598496",
            "extra": "mean: 72.00663949839132 usec\nrounds: 11165"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1283.0663280299098,
            "unit": "iter/sec",
            "range": "stddev: 0.00011276892168841242",
            "extra": "mean: 779.3829345794264 usec\nrounds: 963"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1306.1352592346157,
            "unit": "iter/sec",
            "range": "stddev: 0.000013283431227368041",
            "extra": "mean: 765.617490937341 usec\nrounds: 1269"
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
          "id": "87dc7ca5051f1c191982b80c0e41789f3a791d81",
          "message": "fix: CSR build uses local SSD temp dir instead of slow external storage\n\nbuild_csr_from_pending() now writes mmap files to a temp dir on the\nlocal SSD (fast random writes) instead of the external data_dir\nwhich may be on slow USB storage. Files are moved on save.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T08:22:56+02:00",
          "tree_id": "2fa17605ea5c2ee9d2a309d823a4c3863cb8ebff",
          "url": "https://github.com/kkollsga/kglite/commit/87dc7ca5051f1c191982b80c0e41789f3a791d81"
        },
        "date": 1775370309872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1150.9676701917092,
            "unit": "iter/sec",
            "range": "stddev: 0.000022519092078691377",
            "extra": "mean: 868.8341348749062 usec\nrounds: 519"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 819.5000899061632,
            "unit": "iter/sec",
            "range": "stddev: 0.00007182169055581058",
            "extra": "mean: 1.2202561199407616 msec\nrounds: 667"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13868.16799024065,
            "unit": "iter/sec",
            "range": "stddev: 0.000002694514899692683",
            "extra": "mean: 72.1075776341708 usec\nrounds: 6653"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1505.0116418880882,
            "unit": "iter/sec",
            "range": "stddev: 0.00002663944875505932",
            "extra": "mean: 664.4466874325743 usec\nrounds: 931"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 655720.787821057,
            "unit": "iter/sec",
            "range": "stddev: 3.268471777658883e-7",
            "extra": "mean: 1.5250393438386693 usec\nrounds: 79733"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128629.28023239553,
            "unit": "iter/sec",
            "range": "stddev: 7.074401080944917e-7",
            "extra": "mean: 7.77427968339162 usec\nrounds: 20466"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2683.3686390736757,
            "unit": "iter/sec",
            "range": "stddev: 0.000007960345360876973",
            "extra": "mean: 372.6659041320575 usec\nrounds: 3630"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1476.0360313753486,
            "unit": "iter/sec",
            "range": "stddev: 0.0003540698467246127",
            "extra": "mean: 677.490236514223 usec\nrounds: 1205"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13412.92638726299,
            "unit": "iter/sec",
            "range": "stddev: 0.000002535002078223722",
            "extra": "mean: 74.55494581328703 usec\nrounds: 8563"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1438.2245251504253,
            "unit": "iter/sec",
            "range": "stddev: 0.00003795652658768905",
            "extra": "mean: 695.3017296763237 usec\nrounds: 1021"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1442.8548565683761,
            "unit": "iter/sec",
            "range": "stddev: 0.00009089590925215067",
            "extra": "mean: 693.0704051399576 usec\nrounds: 1323"
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
          "id": "d669994251a0b0440b2e880bc0020c7d4eae9644",
          "message": "fix: eliminate 39 GB pre-fill I/O in CSR build\n\nbuild_csr_from_pending() was pre-filling three 13 GB mmap arrays with\nzeros before writing actual data — 78 GB total I/O causing 1+ hour\nstall. Fix: use mapped_prefilled() which sets file size but lets the\nOS lazy-zero-fill pages. Only pages actually written get I/O.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T09:45:34+02:00",
          "tree_id": "d91944b1e97f999a88aeded8e1b0195ab21d117d",
          "url": "https://github.com/kkollsga/kglite/commit/d669994251a0b0440b2e880bc0020c7d4eae9644"
        },
        "date": 1775375267189,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1084.5941066470762,
            "unit": "iter/sec",
            "range": "stddev: 0.000017580336451081074",
            "extra": "mean: 922.0039034615528 usec\nrounds: 549"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 736.987074097421,
            "unit": "iter/sec",
            "range": "stddev: 0.00003397150847387292",
            "extra": "mean: 1.3568759007404407 msec\nrounds: 675"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13707.18327992759,
            "unit": "iter/sec",
            "range": "stddev: 0.000005090872365626519",
            "extra": "mean: 72.95444874253425 usec\nrounds: 7316"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1542.4607629697912,
            "unit": "iter/sec",
            "range": "stddev: 0.000035842294643775985",
            "extra": "mean: 648.3147085535197 usec\nrounds: 947"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 693728.2537930664,
            "unit": "iter/sec",
            "range": "stddev: 4.7542354581914015e-7",
            "extra": "mean: 1.4414866261138215 usec\nrounds: 76081"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128293.01077012953,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013560078057891366",
            "extra": "mean: 7.794656887363579 usec\nrounds: 31870"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2506.272779113534,
            "unit": "iter/sec",
            "range": "stddev: 0.00003277736299204433",
            "extra": "mean: 398.99886729556187 usec\nrounds: 4559"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1514.685582638982,
            "unit": "iter/sec",
            "range": "stddev: 0.00002309696412391131",
            "extra": "mean: 660.2030226350581 usec\nrounds: 1237"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13615.341254321032,
            "unit": "iter/sec",
            "range": "stddev: 0.000004954783923078134",
            "extra": "mean: 73.44656158967996 usec\nrounds: 11122"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1271.5096231198559,
            "unit": "iter/sec",
            "range": "stddev: 0.000028123322224752602",
            "extra": "mean: 786.4667178423213 usec\nrounds: 964"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1286.0375581786982,
            "unit": "iter/sec",
            "range": "stddev: 0.000013812045988779538",
            "extra": "mean: 777.5822670499703 usec\nrounds: 1217"
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
          "id": "a07e0232b83ccd59f2c9a51aefeef9f8e9daf23a",
          "message": "fix: sequential CSR writes with sort_indices (no page faults)\n\nReplace random-access mmap set() with sort-then-push() pattern.\nsort_indices Vec is only 4 bytes/edge (3.4 GB for 862M edges)\nvs 16 bytes/edge for the indexed Vec. Fits in 16 GB RAM.\n\nSequential writes eliminate mmap page faults that caused Phase 3\nto stall for 1+ hour on 862M edges.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T10:49:23+02:00",
          "tree_id": "cb08a472c6233d5a862b8194cd08be1e294f1e51",
          "url": "https://github.com/kkollsga/kglite/commit/a07e0232b83ccd59f2c9a51aefeef9f8e9daf23a"
        },
        "date": 1775379095882,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1085.572816622127,
            "unit": "iter/sec",
            "range": "stddev: 0.000021206375212413785",
            "extra": "mean: 921.1726608184647 usec\nrounds: 513"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 730.7614501995779,
            "unit": "iter/sec",
            "range": "stddev: 0.00004254729781319163",
            "extra": "mean: 1.3684356224960834 msec\nrounds: 649"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13810.697723279398,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052058277496337695",
            "extra": "mean: 72.4076379077064 usec\nrounds: 7302"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1554.63061677687,
            "unit": "iter/sec",
            "range": "stddev: 0.00009436100320821471",
            "extra": "mean: 643.2396153841643 usec\nrounds: 884"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 694755.7859214729,
            "unit": "iter/sec",
            "range": "stddev: 4.183644058728531e-7",
            "extra": "mean: 1.4393546916254518 usec\nrounds: 87101"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 129724.13071827669,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013456291962296994",
            "extra": "mean: 7.7086660320099645 usec\nrounds: 21535"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2566.195185820667,
            "unit": "iter/sec",
            "range": "stddev: 0.000014663497925772479",
            "extra": "mean: 389.6819717866476 usec\nrounds: 4466"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1569.041999922285,
            "unit": "iter/sec",
            "range": "stddev: 0.00002127731837366303",
            "extra": "mean: 637.331569231117 usec\nrounds: 1365"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13673.832397801914,
            "unit": "iter/sec",
            "range": "stddev: 0.000006047800027039024",
            "extra": "mean: 73.13238680333329 usec\nrounds: 7911"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1238.205660698107,
            "unit": "iter/sec",
            "range": "stddev: 0.00017920738861689595",
            "extra": "mean: 807.6202780693109 usec\nrounds: 953"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1280.8294949018773,
            "unit": "iter/sec",
            "range": "stddev: 0.000017874443116128587",
            "extra": "mean: 780.7440443715021 usec\nrounds: 1217"
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
          "id": "d9a034ba2f64ad73d34dac4ac8e78c76319ccb26",
          "message": "fix: free pending_edges before sort to avoid swap thrashing\n\nPhase 3 CSR build was swap-bound: 10 GB pending_edges + 3.4 GB\nsort_indices = 13.4 GB on 16 GB machine → heavy swapping.\nFix: build edge_endpoints first, drop pending_edges (frees 10 GB),\nthen sort using edge_endpoints for lookups. Peak: ~3.4 GB.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T11:47:26+02:00",
          "tree_id": "876c9cd861c5a67ba534ee8a2f5a439d729e6ff0",
          "url": "https://github.com/kkollsga/kglite/commit/d9a034ba2f64ad73d34dac4ac8e78c76319ccb26"
        },
        "date": 1775382576717,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1068.4177332275863,
            "unit": "iter/sec",
            "range": "stddev: 0.0001589465994030825",
            "extra": "mean: 935.9634990137212 usec\nrounds: 507"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 749.90679283498,
            "unit": "iter/sec",
            "range": "stddev: 0.00002756384162573359",
            "extra": "mean: 1.333499055555367 msec\nrounds: 684"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13747.95832308115,
            "unit": "iter/sec",
            "range": "stddev: 0.000004893836785502791",
            "extra": "mean: 72.73807328329777 usec\nrounds: 7587"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1562.4002338851822,
            "unit": "iter/sec",
            "range": "stddev: 0.00006208208493753286",
            "extra": "mean: 640.040866809988 usec\nrounds: 931"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 693861.8706737603,
            "unit": "iter/sec",
            "range": "stddev: 3.957336319541386e-7",
            "extra": "mean: 1.4412090392414423 usec\nrounds: 117289"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131072.71755932824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011683672602862907",
            "extra": "mean: 7.629352764028593 usec\nrounds: 30210"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2592.049246976033,
            "unit": "iter/sec",
            "range": "stddev: 0.00001734819903527232",
            "extra": "mean: 385.795139180566 usec\nrounds: 4857"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1541.6837370319577,
            "unit": "iter/sec",
            "range": "stddev: 0.00009461761932824996",
            "extra": "mean: 648.6414664561457 usec\nrounds: 1267"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14162.74841901008,
            "unit": "iter/sec",
            "range": "stddev: 0.00000451620186701468",
            "extra": "mean: 70.60776414398076 usec\nrounds: 9527"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1293.1025762026052,
            "unit": "iter/sec",
            "range": "stddev: 0.00009909621138628784",
            "extra": "mean: 773.3338548722515 usec\nrounds: 944"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1293.7151583564153,
            "unit": "iter/sec",
            "range": "stddev: 0.000016024708287201135",
            "extra": "mean: 772.9676764941348 usec\nrounds: 1255"
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
          "id": "93ee9434efb90ddcaf98d09782df7a025aed6320",
          "message": "fix: CSR build stays within 13.4 GB peak (sort + mmap sequential push)\n\nSort pending_edges in-place (no extra copy), write CSR arrays via\nmmap sequential push (fast on any storage). Peak: pending (10 GB) +\nsort_indices (3.4 GB) = 13.4 GB. Fits in 16 GB RAM without swap.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-05T12:36:39+02:00",
          "tree_id": "576481b323f24f14cfae4a82829fb2eb18dc70e4",
          "url": "https://github.com/kkollsga/kglite/commit/93ee9434efb90ddcaf98d09782df7a025aed6320"
        },
        "date": 1775385526422,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1043.4451887537791,
            "unit": "iter/sec",
            "range": "stddev: 0.000019073367289298237",
            "extra": "mean: 958.3637078190307 usec\nrounds: 486"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 723.2640146363569,
            "unit": "iter/sec",
            "range": "stddev: 0.000035510975801155655",
            "extra": "mean: 1.3826209790110744 msec\nrounds: 667"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12339.692995733885,
            "unit": "iter/sec",
            "range": "stddev: 0.000004807681862937623",
            "extra": "mean: 81.03929330703146 usec\nrounds: 6604"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1515.2411821324242,
            "unit": "iter/sec",
            "range": "stddev: 0.000021161635329589542",
            "extra": "mean: 659.9609433744952 usec\nrounds: 883"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 688987.608228445,
            "unit": "iter/sec",
            "range": "stddev: 4.2250412174415524e-7",
            "extra": "mean: 1.4514049136112093 usec\nrounds: 86942"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127370.88083695745,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014031356921000111",
            "extra": "mean: 7.85108804641197 usec\nrounds: 20387"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2586.7552797948106,
            "unit": "iter/sec",
            "range": "stddev: 0.000040404833573464074",
            "extra": "mean: 386.584694660147 usec\nrounds: 3989"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1555.9985716518963,
            "unit": "iter/sec",
            "range": "stddev: 0.00002100009981017439",
            "extra": "mean: 642.6741118009954 usec\nrounds: 1288"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13652.749229611503,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046482361103827705",
            "extra": "mean: 73.24532101059148 usec\nrounds: 10252"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1194.6022545632538,
            "unit": "iter/sec",
            "range": "stddev: 0.00019243879103744992",
            "extra": "mean: 837.0987047614435 usec\nrounds: 840"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1255.607567765939,
            "unit": "iter/sec",
            "range": "stddev: 0.000016118985613558437",
            "extra": "mean: 796.4271844739411 usec\nrounds: 1198"
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
          "id": "9c13e73590bfa3a4d6aaf7a9e72926f907cb01fa",
          "message": "disk graph fixes",
          "timestamp": "2026-04-05T21:12:50+02:00",
          "tree_id": "f613df4f41ad47bcbbb75a54f2f52717e4a67b65",
          "url": "https://github.com/kkollsga/kglite/commit/9c13e73590bfa3a4d6aaf7a9e72926f907cb01fa"
        },
        "date": 1775416506872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1056.4416173601114,
            "unit": "iter/sec",
            "range": "stddev: 0.00015622564615966965",
            "extra": "mean: 946.5738414384407 usec\nrounds: 473"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 747.8623298283002,
            "unit": "iter/sec",
            "range": "stddev: 0.00003082395971909099",
            "extra": "mean: 1.3371444985463936 msec\nrounds: 688"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13519.125250996858,
            "unit": "iter/sec",
            "range": "stddev: 0.000005126400430338712",
            "extra": "mean: 73.96928288139524 usec\nrounds: 7010"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1544.3499324335473,
            "unit": "iter/sec",
            "range": "stddev: 0.000024420138785453636",
            "extra": "mean: 647.5216393633181 usec\nrounds: 879"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 685231.3007077824,
            "unit": "iter/sec",
            "range": "stddev: 4.700492818687207e-7",
            "extra": "mean: 1.459361238996365 usec\nrounds: 112020"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131042.50098894666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013516098789468096",
            "extra": "mean: 7.631111986212392 usec\nrounds: 21708"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2542.671662603748,
            "unit": "iter/sec",
            "range": "stddev: 0.00001653616024281458",
            "extra": "mean: 393.2871139862312 usec\nrounds: 4483"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1524.6329699462688,
            "unit": "iter/sec",
            "range": "stddev: 0.000030720038763937526",
            "extra": "mean: 655.8955628745467 usec\nrounds: 1169"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13908.116194553162,
            "unit": "iter/sec",
            "range": "stddev: 0.000005176255529055624",
            "extra": "mean: 71.90046344246319 usec\nrounds: 11311"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1164.6788707936425,
            "unit": "iter/sec",
            "range": "stddev: 0.00019036824864639205",
            "extra": "mean: 858.6057711500972 usec\nrounds: 922"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1164.4517784994575,
            "unit": "iter/sec",
            "range": "stddev: 0.000020666406057233033",
            "extra": "mean: 858.7732171173508 usec\nrounds: 1110"
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
          "id": "ab36aaa022f2c3ac8c2e7efde53d913153c34e8a",
          "message": "DiskGraph performance improvements",
          "timestamp": "2026-04-06T01:53:35+02:00",
          "tree_id": "6c39e4c96678da2b1dddbde1dcd82e3204318e5b",
          "url": "https://github.com/kkollsga/kglite/commit/ab36aaa022f2c3ac8c2e7efde53d913153c34e8a"
        },
        "date": 1775433351009,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1022.321718386492,
            "unit": "iter/sec",
            "range": "stddev: 0.00012998516589679987",
            "extra": "mean: 978.1656615671611 usec\nrounds: 523"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 722.6700657003677,
            "unit": "iter/sec",
            "range": "stddev: 0.000032644315201323004",
            "extra": "mean: 1.3837573291912417 msec\nrounds: 644"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 11974.619534362102,
            "unit": "iter/sec",
            "range": "stddev: 0.000014263708340016476",
            "extra": "mean: 83.50996013947852 usec\nrounds: 7175"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1541.156982943511,
            "unit": "iter/sec",
            "range": "stddev: 0.000020973409721952198",
            "extra": "mean: 648.8631664829264 usec\nrounds: 901"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 685766.4708760758,
            "unit": "iter/sec",
            "range": "stddev: 4.384622781647471e-7",
            "extra": "mean: 1.458222357710908 usec\nrounds: 116605"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127766.42017273942,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012266681912575558",
            "extra": "mean: 7.826782644829574 usec\nrounds: 33857"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2584.262216549802,
            "unit": "iter/sec",
            "range": "stddev: 0.00004440477699358486",
            "extra": "mean: 386.9576367273908 usec\nrounds: 4253"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1578.3208799412525,
            "unit": "iter/sec",
            "range": "stddev: 0.000024059401105202842",
            "extra": "mean: 633.5847245695828 usec\nrounds: 1278"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13847.877176300557,
            "unit": "iter/sec",
            "range": "stddev: 0.000004354845910749534",
            "extra": "mean: 72.21323436572743 usec\nrounds: 11401"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1257.6753678022708,
            "unit": "iter/sec",
            "range": "stddev: 0.000096237869890216",
            "extra": "mean: 795.1177431004738 usec\nrounds: 942"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1257.3239822452335,
            "unit": "iter/sec",
            "range": "stddev: 0.0000855632683478214",
            "extra": "mean: 795.3399554300047 usec\nrounds: 1234"
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
          "id": "ce5171e5480b746ac562e92ef9c9ceb850136056",
          "message": "fix",
          "timestamp": "2026-04-06T19:38:11+02:00",
          "tree_id": "013de5bd5c4ac8a0678d8686a30576f4924e073e",
          "url": "https://github.com/kkollsga/kglite/commit/ce5171e5480b746ac562e92ef9c9ceb850136056"
        },
        "date": 1775497228167,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1045.2718142887445,
            "unit": "iter/sec",
            "range": "stddev: 0.000023073402925548182",
            "extra": "mean: 956.6889552842772 usec\nrounds: 492"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 734.0744133685073,
            "unit": "iter/sec",
            "range": "stddev: 0.000028308959503058564",
            "extra": "mean: 1.3622597134413366 msec\nrounds: 677"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12539.858366787494,
            "unit": "iter/sec",
            "range": "stddev: 0.000006466487265354547",
            "extra": "mean: 79.74571727608624 usec\nrounds: 6975"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1491.6033827883111,
            "unit": "iter/sec",
            "range": "stddev: 0.00006956101589310596",
            "extra": "mean: 670.4195039640242 usec\nrounds: 883"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 686543.2465249727,
            "unit": "iter/sec",
            "range": "stddev: 4.2160901398686654e-7",
            "extra": "mean: 1.456572481138849 usec\nrounds: 105186"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 122945.13701283076,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013345794115481507",
            "extra": "mean: 8.133709264935288 usec\nrounds: 25181"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2436.5346238512766,
            "unit": "iter/sec",
            "range": "stddev: 0.00001477980969530367",
            "extra": "mean: 410.41895740408694 usec\nrounds: 3991"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1511.0047820450866,
            "unit": "iter/sec",
            "range": "stddev: 0.000019676730577113647",
            "extra": "mean: 661.8112741155846 usec\nrounds: 1244"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13026.107036033127,
            "unit": "iter/sec",
            "range": "stddev: 0.000013225537614132272",
            "extra": "mean: 76.76890702907447 usec\nrounds: 10229"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1184.2758171488767,
            "unit": "iter/sec",
            "range": "stddev: 0.0007442942947687508",
            "extra": "mean: 844.3978890048455 usec\nrounds: 964"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1280.2776641996545,
            "unit": "iter/sec",
            "range": "stddev: 0.000014329539693903018",
            "extra": "mean: 781.0805639768263 usec\nrounds: 1227"
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
          "id": "038bf6cc0e1d46239e023eb347bdd1a1500e864d",
          "message": "release: v0.7.3 — single-file mmap columns, property log, partitioned CSR\n\nDisk build pipeline rewrite: properties spilled to zstd log during Phase 1,\nreplayed in bulk Phase 1b. Column stores written to single columns.bin with\nmmap-backed reads. CSR switched to hash-partitioned (Kuzu pattern). Pending\nedges buffer file-backed to avoid 14 GB heap at Wikidata scale. Auto-typing\nfrom P31 and sparse property overflow bag for wide schemas.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T13:30:25+02:00",
          "tree_id": "d5e9a80c08a55164d69cef3273a582199074d069",
          "url": "https://github.com/kkollsga/kglite/commit/038bf6cc0e1d46239e023eb347bdd1a1500e864d"
        },
        "date": 1775647970196,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1091.617575427256,
            "unit": "iter/sec",
            "range": "stddev: 0.00004138162398471554",
            "extra": "mean: 916.0717292487736 usec\nrounds: 506"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 719.94307726882,
            "unit": "iter/sec",
            "range": "stddev: 0.000031447011453973836",
            "extra": "mean: 1.3889987022218553 msec\nrounds: 675"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13687.951897798883,
            "unit": "iter/sec",
            "range": "stddev: 0.000005181621225839908",
            "extra": "mean: 73.05694872881654 usec\nrounds: 7119"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1560.8696007549668,
            "unit": "iter/sec",
            "range": "stddev: 0.0000264449867411857",
            "extra": "mean: 640.6685090902639 usec\nrounds: 825"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 710410.2300626345,
            "unit": "iter/sec",
            "range": "stddev: 4.4448025660819914e-7",
            "extra": "mean: 1.4076373870796222 usec\nrounds: 73282"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130109.90637831065,
            "unit": "iter/sec",
            "range": "stddev: 0.000001325843242025225",
            "extra": "mean: 7.685809849807871 usec\nrounds: 26295"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2643.696819672836,
            "unit": "iter/sec",
            "range": "stddev: 0.000012440492609579941",
            "extra": "mean: 378.25819986565347 usec\nrounds: 4468"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1514.498942033816,
            "unit": "iter/sec",
            "range": "stddev: 0.000025480467575293472",
            "extra": "mean: 660.284383333476 usec\nrounds: 1260"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13922.10813208722,
            "unit": "iter/sec",
            "range": "stddev: 0.000004509969496223344",
            "extra": "mean: 71.82820234639843 usec\nrounds: 11421"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1242.2203593383822,
            "unit": "iter/sec",
            "range": "stddev: 0.00005712601138382297",
            "extra": "mean: 805.0101517677659 usec\nrounds: 962"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1257.5834171052707,
            "unit": "iter/sec",
            "range": "stddev: 0.000016448974035922478",
            "extra": "mean: 795.1758797057129 usec\nrounds: 1222"
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
          "id": "f62cfee5fc13a9250113bf45f92b6ad2015ea298",
          "message": "delete",
          "timestamp": "2026-04-08T13:31:00+02:00",
          "tree_id": "8103d65f0e079c0f47bb6420934d4b4aa3675519",
          "url": "https://github.com/kkollsga/kglite/commit/f62cfee5fc13a9250113bf45f92b6ad2015ea298"
        },
        "date": 1775647997650,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1036.0250391977079,
            "unit": "iter/sec",
            "range": "stddev: 0.000025484569357209113",
            "extra": "mean: 965.2276365582771 usec\nrounds: 465"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 716.5037492617801,
            "unit": "iter/sec",
            "range": "stddev: 0.000041199860056271816",
            "extra": "mean: 1.395666109256663 msec\nrounds: 659"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12214.260912048125,
            "unit": "iter/sec",
            "range": "stddev: 0.000005633648746810354",
            "extra": "mean: 81.87151127692071 usec\nrounds: 6429"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1470.0741517132312,
            "unit": "iter/sec",
            "range": "stddev: 0.000023767093780627345",
            "extra": "mean: 680.2377953755566 usec\nrounds: 865"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 679317.9774659598,
            "unit": "iter/sec",
            "range": "stddev: 4.208998986633945e-7",
            "extra": "mean: 1.4720646783561817 usec\nrounds: 67627"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127221.17341870826,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012724083885059082",
            "extra": "mean: 7.860326808249255 usec\nrounds: 21251"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2471.126382357476,
            "unit": "iter/sec",
            "range": "stddev: 0.00001925656330193883",
            "extra": "mean: 404.6737581450575 usec\nrounds: 4205"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1449.6140475822328,
            "unit": "iter/sec",
            "range": "stddev: 0.00009482688511383397",
            "extra": "mean: 689.8387896198091 usec\nrounds: 1079"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13499.999615052482,
            "unit": "iter/sec",
            "range": "stddev: 0.000004805750801090378",
            "extra": "mean: 74.07407618626901 usec\nrounds: 10960"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1239.8404856957793,
            "unit": "iter/sec",
            "range": "stddev: 0.000028685730203054798",
            "extra": "mean: 806.5553686439071 usec\nrounds: 944"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1243.2371342985641,
            "unit": "iter/sec",
            "range": "stddev: 0.000030293558532448345",
            "extra": "mean: 804.3517784434595 usec\nrounds: 1169"
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
          "id": "fda521b03643af86d6791f609172c29f67c853be",
          "message": "optimization",
          "timestamp": "2026-04-08T16:34:07+02:00",
          "tree_id": "610f5d1f33e7f1385ead5652686a9d1c51a8c95c",
          "url": "https://github.com/kkollsga/kglite/commit/fda521b03643af86d6791f609172c29f67c853be"
        },
        "date": 1775659014667,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1051.4301435370796,
            "unit": "iter/sec",
            "range": "stddev: 0.000027993159065664216",
            "extra": "mean: 951.0855344473334 usec\nrounds: 479"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 727.4744399222465,
            "unit": "iter/sec",
            "range": "stddev: 0.000039275178392852945",
            "extra": "mean: 1.374618742765562 msec\nrounds: 622"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12310.390373130815,
            "unit": "iter/sec",
            "range": "stddev: 0.000005880454608019183",
            "extra": "mean: 81.2321924561095 usec\nrounds: 5700"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1506.598281926639,
            "unit": "iter/sec",
            "range": "stddev: 0.000020771274419260497",
            "extra": "mean: 663.7469403729833 usec\nrounds: 805"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 697766.5376000294,
            "unit": "iter/sec",
            "range": "stddev: 4.272002357689006e-7",
            "extra": "mean: 1.4331441049602405 usec\nrounds: 66854"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126081.82912822788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012958711796291211",
            "extra": "mean: 7.93135701563291 usec\nrounds: 20226"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2376.1915672703926,
            "unit": "iter/sec",
            "range": "stddev: 0.00001983497133168824",
            "extra": "mean: 420.84149012814316 usec\nrounds: 3444"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1500.170476896463,
            "unit": "iter/sec",
            "range": "stddev: 0.000027410024787621432",
            "extra": "mean: 666.5909077672221 usec\nrounds: 1236"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13297.709087621299,
            "unit": "iter/sec",
            "range": "stddev: 0.000020163179107335727",
            "extra": "mean: 75.20092321247198 usec\nrounds: 9103"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1249.038235210745,
            "unit": "iter/sec",
            "range": "stddev: 0.00005197581550550375",
            "extra": "mean: 800.6160034254469 usec\nrounds: 876"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1245.5552296495885,
            "unit": "iter/sec",
            "range": "stddev: 0.000018885720340285996",
            "extra": "mean: 802.854804183456 usec\nrounds: 1195"
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
          "id": "93a6df65f949c5f788c17692f514c7e85ff0723b",
          "message": "release: v0.7.4 — CsrEdge 8-byte, query pre-filter, Cypher LIMIT pushdown\n\nDiskGraph storage:\n- CsrEdge shrunk from 16 to 8 bytes (conn_type stored only in EdgeEndpoints)\n- Edge conn_type pre-filter skips materialization for non-matching edges\n- Arena clearing at query boundaries prevents OOM\n- node_type_of() reads mmap'd node_slots without materializing NodeData\n- Edge properties fast path skips HashMap lookup when empty\n\nCypher engine (both backends):\n- Source node cap with LIMIT avoids O(N) allocation on large types\n- expand_from_node stops after collecting enough results\n- WHERE id(n)=X pushed into pattern as {id: X} property\n- Cross-type id lookup for untyped {id: X} patterns\n- estimate_node_selectivity returns 1 for {id: X}\n- id_indices built from column stores on disk graph load (no materialization)\n- lookup_by_id_normalized trusts id_indices without fallback scan\n- has_connection_type falls back to interner when metadata empty\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T21:46:58+02:00",
          "tree_id": "6031ededc14927f8b1b8cf300a397b379f61aa8c",
          "url": "https://github.com/kkollsga/kglite/commit/93a6df65f949c5f788c17692f514c7e85ff0723b"
        },
        "date": 1775677757526,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1062.166567938505,
            "unit": "iter/sec",
            "range": "stddev: 0.000021463213948980803",
            "extra": "mean: 941.4719218105682 usec\nrounds: 486"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 738.5160712900039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000342737340913086",
            "extra": "mean: 1.3540666735298648 msec\nrounds: 680"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12668.198729566291,
            "unit": "iter/sec",
            "range": "stddev: 0.000005030425178397941",
            "extra": "mean: 78.93782070738293 usec\nrounds: 6587"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1576.3950951124752,
            "unit": "iter/sec",
            "range": "stddev: 0.000022817909284442058",
            "extra": "mean: 634.3587360176672 usec\nrounds: 894"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 676207.4083900024,
            "unit": "iter/sec",
            "range": "stddev: 4.2909875830952155e-7",
            "extra": "mean: 1.4788362085250186 usec\nrounds: 85383"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126859.28993618656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013292545151977631",
            "extra": "mean: 7.8827494659872785 usec\nrounds: 21071"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2452.2093866453906,
            "unit": "iter/sec",
            "range": "stddev: 0.000014784138298969112",
            "extra": "mean: 407.79551919422136 usec\nrounds: 4220"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1562.3535415520882,
            "unit": "iter/sec",
            "range": "stddev: 0.000031160603060177655",
            "extra": "mean: 640.0599950038007 usec\nrounds: 1201"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12908.290193688897,
            "unit": "iter/sec",
            "range": "stddev: 0.000004692194061315277",
            "extra": "mean: 77.46959395822373 usec\nrounds: 9600"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1223.992551305725,
            "unit": "iter/sec",
            "range": "stddev: 0.000241729811548945",
            "extra": "mean: 816.9984359244871 usec\nrounds: 952"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1284.4194395185004,
            "unit": "iter/sec",
            "range": "stddev: 0.00006666979318316893",
            "extra": "mean: 778.5618694582178 usec\nrounds: 1218"
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
          "id": "cb4ce3af9650dc1b371581bcabc165c45692a54e",
          "message": "feat: persist id_indices to disk, load without rebuild\n\n- TypeIdIndex now derives Serialize/Deserialize\n- ntriples build saves id_indices to id_indices.bin.zst (bincode + zstd)\n- load() reads from file instead of rebuilding from column stores\n- Fallback: rebuilds from columns if file missing or corrupt\n- Expected to restore load time from ~27s back to ~8s\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T21:51:55+02:00",
          "tree_id": "ca602a2b7528ab33636954cc32506e41c858609a",
          "url": "https://github.com/kkollsga/kglite/commit/cb4ce3af9650dc1b371581bcabc165c45692a54e"
        },
        "date": 1775678062444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1090.1780488300283,
            "unit": "iter/sec",
            "range": "stddev: 0.000019244609587590814",
            "extra": "mean: 917.2813569977797 usec\nrounds: 493"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 749.9878258746742,
            "unit": "iter/sec",
            "range": "stddev: 0.000030559907320090665",
            "extra": "mean: 1.3333549765741182 msec\nrounds: 683"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13347.811341898429,
            "unit": "iter/sec",
            "range": "stddev: 0.000004906296997508925",
            "extra": "mean: 74.91864953627464 usec\nrounds: 7005"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1641.5207597166134,
            "unit": "iter/sec",
            "range": "stddev: 0.000021880935466242914",
            "extra": "mean: 609.19119912479 usec\nrounds: 914"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 696307.8605844894,
            "unit": "iter/sec",
            "range": "stddev: 4.1312218573560547e-7",
            "extra": "mean: 1.4361463608361216 usec\nrounds: 87093"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126602.65692289358,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013977527743462844",
            "extra": "mean: 7.89872838615893 usec\nrounds: 22451"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2525.4084920745613,
            "unit": "iter/sec",
            "range": "stddev: 0.000012039236206097255",
            "extra": "mean: 395.97554341734417 usec\nrounds: 3927"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1613.5542460166268,
            "unit": "iter/sec",
            "range": "stddev: 0.000021631994314666037",
            "extra": "mean: 619.7498488003703 usec\nrounds: 1250"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13342.422012728464,
            "unit": "iter/sec",
            "range": "stddev: 0.000005114810381751633",
            "extra": "mean: 74.94891100326579 usec\nrounds: 10933"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1176.7625057620116,
            "unit": "iter/sec",
            "range": "stddev: 0.00003348165420344539",
            "extra": "mean: 849.7891419071436 usec\nrounds: 902"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1112.1364289460973,
            "unit": "iter/sec",
            "range": "stddev: 0.00015044661847216654",
            "extra": "mean: 899.170258227795 usec\nrounds: 1185"
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
          "id": "42ef8f7a01578fa0da1d25780c2a665dc2e8cf1f",
          "message": "fix: remove id_indices rebuild fallback on load\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T21:54:15+02:00",
          "tree_id": "c07240fe1539005b4364d5f13105c565783fb4e5",
          "url": "https://github.com/kkollsga/kglite/commit/42ef8f7a01578fa0da1d25780c2a665dc2e8cf1f"
        },
        "date": 1775678191268,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 994.4153633200651,
            "unit": "iter/sec",
            "range": "stddev: 0.000022684217482031175",
            "extra": "mean: 1.0056159999995269 msec\nrounds: 459"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 710.8437451903626,
            "unit": "iter/sec",
            "range": "stddev: 0.00014801024105379635",
            "extra": "mean: 1.4067789254194 msec\nrounds: 657"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12506.0346022449,
            "unit": "iter/sec",
            "range": "stddev: 0.000005703235092696672",
            "extra": "mean: 79.96139718184489 usec\nrounds: 6458"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1483.3970445097648,
            "unit": "iter/sec",
            "range": "stddev: 0.000021239903405690118",
            "extra": "mean: 674.1283486448374 usec\nrounds: 849"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 682906.5218185065,
            "unit": "iter/sec",
            "range": "stddev: 4.1380149032323944e-7",
            "extra": "mean: 1.4643292574466964 usec\nrounds: 73769"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127165.55473528842,
            "unit": "iter/sec",
            "range": "stddev: 0.000001288502653216103",
            "extra": "mean: 7.863764696985984 usec\nrounds: 20055"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2436.1669561254052,
            "unit": "iter/sec",
            "range": "stddev: 0.00003989786892540706",
            "extra": "mean: 410.4808980704866 usec\nrounds: 4405"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1441.0623683646206,
            "unit": "iter/sec",
            "range": "stddev: 0.000023409052121057",
            "extra": "mean: 693.9324917177894 usec\nrounds: 1147"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13408.889918590561,
            "unit": "iter/sec",
            "range": "stddev: 0.000014499234340719914",
            "extra": "mean: 74.57738903602784 usec\nrounds: 9814"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1218.0570356995731,
            "unit": "iter/sec",
            "range": "stddev: 0.00023911394983863242",
            "extra": "mean: 820.9796181060314 usec\nrounds: 961"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1270.3101560609039,
            "unit": "iter/sec",
            "range": "stddev: 0.00004235642840821325",
            "extra": "mean: 787.2093246116312 usec\nrounds: 1223"
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
          "id": "240cb845b22b878d67a6182a2458351de610d2b2",
          "message": "perf: persist type_indices + binary columns_meta for fast load\n\n- type_indices saved as bincode+zstd during build, loaded on disk graph open\n  (eliminates 134M node_slots scan, ~5s → <1s)\n- columns_meta saved as bincode+zstd alongside JSON\n  (load prefers binary, ~3s JSON parse → <0.5s bincode deserialize)\n- JSON files kept for backward compatibility and human readability\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T21:57:56+02:00",
          "tree_id": "5d76481e3fc5e72ca40e8eb69010ec63e978feba",
          "url": "https://github.com/kkollsga/kglite/commit/240cb845b22b878d67a6182a2458351de610d2b2"
        },
        "date": 1775678518149,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1026.2490237300904,
            "unit": "iter/sec",
            "range": "stddev: 0.000027921710983478935",
            "extra": "mean: 974.4223642379864 usec\nrounds: 453"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 712.6580561151313,
            "unit": "iter/sec",
            "range": "stddev: 0.000031875862622544104",
            "extra": "mean: 1.403197496217524 msec\nrounds: 661"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12375.067671633693,
            "unit": "iter/sec",
            "range": "stddev: 0.000005133534609167086",
            "extra": "mean: 80.80763891838865 usec\nrounds: 6655"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1583.705223391913,
            "unit": "iter/sec",
            "range": "stddev: 0.000022739313973577562",
            "extra": "mean: 631.430638246076 usec\nrounds: 821"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 690488.9477413081,
            "unit": "iter/sec",
            "range": "stddev: 4.035896803098816e-7",
            "extra": "mean: 1.4482491041618384 usec\nrounds: 86791"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128726.5415289439,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012875952179458778",
            "extra": "mean: 7.768405708120046 usec\nrounds: 23300"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2453.646106613568,
            "unit": "iter/sec",
            "range": "stddev: 0.000016773369396271838",
            "extra": "mean: 407.55673660704196 usec\nrounds: 4704"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1524.2029698449685,
            "unit": "iter/sec",
            "range": "stddev: 0.00007718075568924115",
            "extra": "mean: 656.0806006707317 usec\nrounds: 1192"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12656.05576443561,
            "unit": "iter/sec",
            "range": "stddev: 0.000005321755350168793",
            "extra": "mean: 79.01355830068866 usec\nrounds: 9957"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1204.4719415484162,
            "unit": "iter/sec",
            "range": "stddev: 0.00032479658744991957",
            "extra": "mean: 830.2393484687105 usec\nrounds: 947"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1248.6242898584846,
            "unit": "iter/sec",
            "range": "stddev: 0.00009897663238616262",
            "extra": "mean: 800.8814245583329 usec\nrounds: 1246"
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
          "id": "75fa87279c8c811954de9a0bdb09b00215061b62",
          "message": "organization",
          "timestamp": "2026-04-10T09:12:18+02:00",
          "tree_id": "bbf74deadad5378d16a75f2f5dd932e03a833f50",
          "url": "https://github.com/kkollsga/kglite/commit/75fa87279c8c811954de9a0bdb09b00215061b62"
        },
        "date": 1775805296965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1014.4799430420858,
            "unit": "iter/sec",
            "range": "stddev: 0.00003324252821249866",
            "extra": "mean: 985.7267330503693 usec\nrounds: 472"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 713.8680947740571,
            "unit": "iter/sec",
            "range": "stddev: 0.00004130998616872318",
            "extra": "mean: 1.4008190130930354 msec\nrounds: 611"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12337.977841039485,
            "unit": "iter/sec",
            "range": "stddev: 0.000021045770874222243",
            "extra": "mean: 81.05055892333725 usec\nrounds: 5015"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1614.5041081175102,
            "unit": "iter/sec",
            "range": "stddev: 0.00006632466471715565",
            "extra": "mean: 619.3852310267494 usec\nrounds: 896"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 676024.232948241,
            "unit": "iter/sec",
            "range": "stddev: 4.4944646247122526e-7",
            "extra": "mean: 1.479236913799455 usec\nrounds: 99513"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126037.65607351821,
            "unit": "iter/sec",
            "range": "stddev: 0.000001450752569824023",
            "extra": "mean: 7.934136758435878 usec\nrounds: 18763"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2533.2559797840972,
            "unit": "iter/sec",
            "range": "stddev: 0.00001390210214625499",
            "extra": "mean: 394.74889548478535 usec\nrounds: 3588"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1597.1888477159118,
            "unit": "iter/sec",
            "range": "stddev: 0.00002288789789439322",
            "extra": "mean: 626.1000390968593 usec\nrounds: 1151"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13409.070341975703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067548973782604575",
            "extra": "mean: 74.57638557310001 usec\nrounds: 7791"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1189.6493389997675,
            "unit": "iter/sec",
            "range": "stddev: 0.00023034735958871237",
            "extra": "mean: 840.5838319052733 usec\nrounds: 934"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1242.0949601946068,
            "unit": "iter/sec",
            "range": "stddev: 0.000016683115127979937",
            "extra": "mean: 805.0914238017066 usec\nrounds: 1168"
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
          "id": "02a6bdd42a38d9ca9c0dd51a16a9688893eeb803",
          "message": "fix",
          "timestamp": "2026-04-12T16:15:27+02:00",
          "tree_id": "8a73badc19c27b3ed5eee36731231237a6a4e432",
          "url": "https://github.com/kkollsga/kglite/commit/02a6bdd42a38d9ca9c0dd51a16a9688893eeb803"
        },
        "date": 1776003472059,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1042.8967222167507,
            "unit": "iter/sec",
            "range": "stddev: 0.00002322423086354899",
            "extra": "mean: 958.8677178641708 usec\nrounds: 599"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 733.4679495124632,
            "unit": "iter/sec",
            "range": "stddev: 0.000027553188360524717",
            "extra": "mean: 1.363386090237073 msec\nrounds: 676"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12201.238960992981,
            "unit": "iter/sec",
            "range": "stddev: 0.000004841145621409499",
            "extra": "mean: 81.95888984692226 usec\nrounds: 7072"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1610.454384135589,
            "unit": "iter/sec",
            "range": "stddev: 0.000021329308261230898",
            "extra": "mean: 620.9427661229596 usec\nrounds: 915"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 682848.537192292,
            "unit": "iter/sec",
            "range": "stddev: 4.26227655545808e-7",
            "extra": "mean: 1.464453602129336 usec\nrounds: 125566"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127350.99051089598,
            "unit": "iter/sec",
            "range": "stddev: 0.000001222849344146732",
            "extra": "mean: 7.852314269314156 usec\nrounds: 31018"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2522.864017497508,
            "unit": "iter/sec",
            "range": "stddev: 0.00004070347976157283",
            "extra": "mean: 396.37491084118165 usec\nrounds: 4206"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1630.0342417098934,
            "unit": "iter/sec",
            "range": "stddev: 0.000019626855189388148",
            "extra": "mean: 613.4840449431342 usec\nrounds: 1157"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13453.96879841925,
            "unit": "iter/sec",
            "range": "stddev: 0.000004637548883793511",
            "extra": "mean: 74.3275099699572 usec\nrounds: 8726"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1045.8907842078322,
            "unit": "iter/sec",
            "range": "stddev: 0.00009096535747483335",
            "extra": "mean: 956.1227760099346 usec\nrounds: 817"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1032.9652786704687,
            "unit": "iter/sec",
            "range": "stddev: 0.000021971593558665706",
            "extra": "mean: 968.0867504928158 usec\nrounds: 1014"
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
          "id": "427fcd99d67c249e16a3b4d2639258a7e6e54993",
          "message": "chore: fix ruff lint errors in benchmark files\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-12T16:24:02+02:00",
          "tree_id": "b7fd2fa8bb61a3f3516aac5dd0dc3bbb30513c8e",
          "url": "https://github.com/kkollsga/kglite/commit/427fcd99d67c249e16a3b4d2639258a7e6e54993"
        },
        "date": 1776003983283,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1058.536712380162,
            "unit": "iter/sec",
            "range": "stddev: 0.000020233921615814556",
            "extra": "mean: 944.7003474744491 usec\nrounds: 495"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 733.13366882352,
            "unit": "iter/sec",
            "range": "stddev: 0.00003199663167631069",
            "extra": "mean: 1.3640077417324563 msec\nrounds: 635"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12505.288232258972,
            "unit": "iter/sec",
            "range": "stddev: 0.000005130244908255543",
            "extra": "mean: 79.96616962577268 usec\nrounds: 6815"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1582.38602993379,
            "unit": "iter/sec",
            "range": "stddev: 0.000020349632150571097",
            "extra": "mean: 631.9570452993963 usec\nrounds: 883"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 670886.9962473119,
            "unit": "iter/sec",
            "range": "stddev: 4.5757728090453905e-7",
            "extra": "mean: 1.4905639930325398 usec\nrounds: 111895"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 124011.55260415308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000844577232485009",
            "extra": "mean: 8.063764859004841 usec\nrounds: 20358"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2606.007008270127,
            "unit": "iter/sec",
            "range": "stddev: 0.000012922659333458649",
            "extra": "mean: 383.7288222274591 usec\nrounds: 4157"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1623.246094053659,
            "unit": "iter/sec",
            "range": "stddev: 0.00002224435078852847",
            "extra": "mean: 616.0495341176182 usec\nrounds: 1275"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13432.935961399317,
            "unit": "iter/sec",
            "range": "stddev: 0.0000049257203977530465",
            "extra": "mean: 74.44388947238228 usec\nrounds: 11047"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1015.4774550556939,
            "unit": "iter/sec",
            "range": "stddev: 0.00004445756867869715",
            "extra": "mean: 984.7584454202926 usec\nrounds: 797"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1022.5345914794073,
            "unit": "iter/sec",
            "range": "stddev: 0.000015338689251894747",
            "extra": "mean: 977.9620252779868 usec\nrounds: 989"
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
          "id": "c784c3dd78c03b8073ad12bb951ec16cae235176",
          "message": "chore: add Cypher scalability benchmarks\n\n- benchmark_cypher_scalability.py: 60 queries across 3 in-memory graph sizes (regression gate)\n- benchmark_wikidata_cypher.py: 45 queries on full Wikidata disk graph (tiered easy→hard)\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T19:16:43+02:00",
          "tree_id": "6274d9bf3c6948eb1d4f2bef577f44b3da18a61d",
          "url": "https://github.com/kkollsga/kglite/commit/c784c3dd78c03b8073ad12bb951ec16cae235176"
        },
        "date": 1776273594377,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1087.9108205799184,
            "unit": "iter/sec",
            "range": "stddev: 0.000035699608740527426",
            "extra": "mean: 919.192989979586 usec\nrounds: 499"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 722.9006757114648,
            "unit": "iter/sec",
            "range": "stddev: 0.0001480061040754933",
            "extra": "mean: 1.3833159016151415 msec\nrounds: 681"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13350.163864687163,
            "unit": "iter/sec",
            "range": "stddev: 0.000008457592490592738",
            "extra": "mean: 74.90544761365243 usec\nrounds: 7082"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1595.7659397239531,
            "unit": "iter/sec",
            "range": "stddev: 0.00002181399235864836",
            "extra": "mean: 626.6583181822938 usec\nrounds: 880"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 701732.6870961363,
            "unit": "iter/sec",
            "range": "stddev: 4.790623472687594e-7",
            "extra": "mean: 1.425044063627895 usec\nrounds: 80815"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 125062.28713313848,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013110569047846053",
            "extra": "mean: 7.996015608889534 usec\nrounds: 20181"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2455.7130562705,
            "unit": "iter/sec",
            "range": "stddev: 0.000022586595015934212",
            "extra": "mean: 407.2137000886836 usec\nrounds: 4508"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1556.1381871429346,
            "unit": "iter/sec",
            "range": "stddev: 0.00002267344465496855",
            "extra": "mean: 642.6164515864733 usec\nrounds: 1198"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13437.14628940646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053620862085654485",
            "extra": "mean: 74.42056359752347 usec\nrounds: 10307"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1020.7050734769115,
            "unit": "iter/sec",
            "range": "stddev: 0.0001837343604994558",
            "extra": "mean: 979.7149303800538 usec\nrounds: 790"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1011.4577338319958,
            "unit": "iter/sec",
            "range": "stddev: 0.0001329971965057649",
            "extra": "mean: 988.6720587042356 usec\nrounds: 988"
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
          "id": "6db9529a693b76119d8d575c2dc102aae601e08f",
          "message": "release: v0.7.9 — zero-allocation counting, WHERE-MATCH fusion, type merge fix\n\nPerformance:\n- Edge count queries 63x faster (zero-allocation CSR counting)\n- WHERE-MATCH fusion: inline predicate evaluation during expansion\n- LIMIT push-down through WHERE clauses\n- Pre-computed edge type counts during CSR build\n\nWikidata pipeline:\n- Q-code type merge now correctly merges indices + column stores\n- Property log key remapping for Phase 1b after type merge\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T20:40:29+02:00",
          "tree_id": "3b35bcb587c366880b85e85ac188d7699c907971",
          "url": "https://github.com/kkollsga/kglite/commit/6db9529a693b76119d8d575c2dc102aae601e08f"
        },
        "date": 1776278625646,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1039.9270185958576,
            "unit": "iter/sec",
            "range": "stddev: 0.00002083780353122762",
            "extra": "mean: 961.6059416844768 usec\nrounds: 463"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 720.338252091283,
            "unit": "iter/sec",
            "range": "stddev: 0.00003062176257205166",
            "extra": "mean: 1.3882367028223257 msec\nrounds: 673"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12439.980729956993,
            "unit": "iter/sec",
            "range": "stddev: 0.000018194321248936702",
            "extra": "mean: 80.38597661103107 usec\nrounds: 5943"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1668.960115320031,
            "unit": "iter/sec",
            "range": "stddev: 0.00002110783633557957",
            "extra": "mean: 599.1754930633829 usec\nrounds: 937"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 672115.5697226534,
            "unit": "iter/sec",
            "range": "stddev: 4.301194165614311e-7",
            "extra": "mean: 1.4878393613358003 usec\nrounds: 83112"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126523.45036613391,
            "unit": "iter/sec",
            "range": "stddev: 0.000001365269712825841",
            "extra": "mean: 7.903673169726222 usec\nrounds: 20913"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2569.3403381890644,
            "unit": "iter/sec",
            "range": "stddev: 0.00001453953109921309",
            "extra": "mean: 389.20495861783155 usec\nrounds: 4978"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1651.5916433304546,
            "unit": "iter/sec",
            "range": "stddev: 0.000022512291036236534",
            "extra": "mean: 605.476543816538 usec\nrounds: 1221"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13646.17995493323,
            "unit": "iter/sec",
            "range": "stddev: 0.00000449944967615581",
            "extra": "mean: 73.2805813277063 usec\nrounds: 11343"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1016.0553597253137,
            "unit": "iter/sec",
            "range": "stddev: 0.00005024578479404818",
            "extra": "mean: 984.1983415847988 usec\nrounds: 808"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1010.9202027311712,
            "unit": "iter/sec",
            "range": "stddev: 0.000017359495746282202",
            "extra": "mean: 989.1977599204482 usec\nrounds: 1008"
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
          "id": "d11fa0a72ce23c7cd459f6a95cdae2e7be5741bf",
          "message": "feat: lightweight peer iteration + String top-K + cold cache prefetch\n\nLightweight peer iteration in expand_from_node:\n- When edge has no named variable and no property filters, uses\n  DiskGraph::iter_peers_filtered() instead of full edge materialization.\n  Skips reading edge_endpoints.bin (13 GB on Wikidata) entirely.\n  Only touches out_edges.bin + node_slots.bin for type checks.\n\nString ORDER BY top-K:\n- FusedOrderByTopK now supports String sort keys via partition-point\n  insertion into a K-element sorted Vec. O(N × K) instead of O(N log N)\n  full sort. For K=20, this is much faster on the sort step (though\n  MATCH materialization still dominates for 13M rows).\n\nCold cache prefetch:\n- MmapOrVec::advise_willneed() calls madvise(WILLNEED) on mmap regions.\n- DiskGraph::prefetch_hot_regions() preloads CSR offset arrays on load.\n- Multi-pattern selectivity reordering in planner.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T21:41:56+02:00",
          "tree_id": "36f8843cdfbe7b71ee4d7de27bbcdb16e8784d01",
          "url": "https://github.com/kkollsga/kglite/commit/d11fa0a72ce23c7cd459f6a95cdae2e7be5741bf"
        },
        "date": 1776282278829,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1132.5297799338207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000240249103064479",
            "extra": "mean: 882.9789889131524 usec\nrounds: 451"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 787.2671386479465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000324997458397479",
            "extra": "mean: 1.2702168691016382 msec\nrounds: 657"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12263.03378213463,
            "unit": "iter/sec",
            "range": "stddev: 0.000004215700106998344",
            "extra": "mean: 81.54588968488757 usec\nrounds: 5303"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1523.3500289346082,
            "unit": "iter/sec",
            "range": "stddev: 0.00004387352979990352",
            "extra": "mean: 656.4479476193493 usec\nrounds: 840"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 653185.7563174233,
            "unit": "iter/sec",
            "range": "stddev: 4.160468378542663e-7",
            "extra": "mean: 1.5309580625852446 usec\nrounds: 100960"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127462.72511306684,
            "unit": "iter/sec",
            "range": "stddev: 0.000001419060083860114",
            "extra": "mean: 7.845430882738008 usec\nrounds: 18172"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2353.3428727533533,
            "unit": "iter/sec",
            "range": "stddev: 0.00003777966925123681",
            "extra": "mean: 424.92745599370505 usec\nrounds: 3704"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1533.9297164332452,
            "unit": "iter/sec",
            "range": "stddev: 0.00008474219634151248",
            "extra": "mean: 651.9203515564194 usec\nrounds: 1189"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13117.901444672008,
            "unit": "iter/sec",
            "range": "stddev: 0.00001018753147016115",
            "extra": "mean: 76.23170552223976 usec\nrounds: 10123"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 937.7441244590913,
            "unit": "iter/sec",
            "range": "stddev: 0.00004819303447932674",
            "extra": "mean: 1.0663889795916546 msec\nrounds: 588"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 947.9689114283524,
            "unit": "iter/sec",
            "range": "stddev: 0.00005550878395735723",
            "extra": "mean: 1.0548869144804018 msec\nrounds: 877"
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
          "id": "bdc906e657f920a110559225aaa0106d0a6a77bb",
          "message": "feat: FusedNodeScanTopK — streaming ORDER BY + LIMIT for node scans\n\nNew fused clause: MATCH (n:Type) RETURN n.prop ORDER BY n.prop LIMIT K\nis executed as a single-pass scan with inline top-K selection. Avoids\nmaterializing all rows — scans nodes directly, evaluates sort key per\nnode, maintains K-element sorted Vec. RETURN expressions are only\nevaluated for the K winners.\n\nPlanner detects: MATCH (single node) [WHERE] RETURN (simple expressions,\nno aggregation, no function calls) ORDER BY (single key) LIMIT K.\n\nResult: \"MATCH (n:human) RETURN n.title ORDER BY n.title LIMIT 20\"\non Wikidata (13M humans) dropped from 22.5s to 15.6s (31% faster).\nRemaining time is I/O-bound (reading title column from 38 GB mmap).\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T21:57:25+02:00",
          "tree_id": "8642b55aac62f10f668dcfe1987f7f1892ae4789",
          "url": "https://github.com/kkollsga/kglite/commit/bdc906e657f920a110559225aaa0106d0a6a77bb"
        },
        "date": 1776283210465,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1130.7330146132524,
            "unit": "iter/sec",
            "range": "stddev: 0.00002539100616692409",
            "extra": "mean: 884.3820663908293 usec\nrounds: 482"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 828.4722213932579,
            "unit": "iter/sec",
            "range": "stddev: 0.00013709693055219462",
            "extra": "mean: 1.207041074133156 msec\nrounds: 634"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13777.08096820614,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028273449739364424",
            "extra": "mean: 72.58431610496706 usec\nrounds: 5340"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1447.8847980303199,
            "unit": "iter/sec",
            "range": "stddev: 0.00018908955981207536",
            "extra": "mean: 690.6626834955271 usec\nrounds: 812"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 636521.0252716347,
            "unit": "iter/sec",
            "range": "stddev: 2.752790375217282e-7",
            "extra": "mean: 1.5710400132866797 usec\nrounds: 113987"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128602.31915563295,
            "unit": "iter/sec",
            "range": "stddev: 6.432460945346929e-7",
            "extra": "mean: 7.77590953697975 usec\nrounds: 18549"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2607.442892157076,
            "unit": "iter/sec",
            "range": "stddev: 0.000009171574955066154",
            "extra": "mean: 383.5175079032023 usec\nrounds: 3859"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1476.4656589987826,
            "unit": "iter/sec",
            "range": "stddev: 0.00015034108432164961",
            "extra": "mean: 677.2930978144915 usec\nrounds: 1053"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13197.141181119252,
            "unit": "iter/sec",
            "range": "stddev: 0.000002797034491689876",
            "extra": "mean: 75.77398667452839 usec\nrounds: 8330"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1094.4217827993011,
            "unit": "iter/sec",
            "range": "stddev: 0.00009243307563850144",
            "extra": "mean: 913.7245034014309 usec\nrounds: 735"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1103.468386847912,
            "unit": "iter/sec",
            "range": "stddev: 0.00006766708181680917",
            "extra": "mean: 906.2334833683163 usec\nrounds: 962"
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
          "id": "addf116771f2a84cc522d9fadc14af9cc46e3138",
          "message": "feat: streaming top-K for FusedMatchReturnAggregate\n\nWhen top_k is set on FusedMatchReturnAggregate, iterate group nodes\ndirectly from type_indices instead of materializing all PatternMatch\nobjects via PatternExecutor. Combined with BinaryHeap, only K elements\nare kept in memory regardless of how many group nodes exist.\n\nThis avoids OOM for typed group nodes (e.g., MATCH (a)-[:P31]->(b:Type)\nRETURN b.title, count(a) ORDER BY ... LIMIT 20). Untyped group nodes\nstill need edge-centric aggregation (future work) since iterating all\n124M nodes with per-node edge counting is O(N × degree).\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T22:02:45+02:00",
          "tree_id": "ddac95fc0cb23daff760319a7303d414fc6f7f01",
          "url": "https://github.com/kkollsga/kglite/commit/addf116771f2a84cc522d9fadc14af9cc46e3138"
        },
        "date": 1776283535187,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1137.790127836573,
            "unit": "iter/sec",
            "range": "stddev: 0.000025229616383252946",
            "extra": "mean: 878.8967099771105 usec\nrounds: 431"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 730.1227770426428,
            "unit": "iter/sec",
            "range": "stddev: 0.000267568062622489",
            "extra": "mean: 1.3696326582913807 msec\nrounds: 597"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 11731.602425815654,
            "unit": "iter/sec",
            "range": "stddev: 0.000016860560216935922",
            "extra": "mean: 85.23984735448225 usec\nrounds: 4933"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1551.1563101147665,
            "unit": "iter/sec",
            "range": "stddev: 0.00002279553133459669",
            "extra": "mean: 644.6803545711085 usec\nrounds: 722"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 640206.742934717,
            "unit": "iter/sec",
            "range": "stddev: 4.5677287285403804e-7",
            "extra": "mean: 1.5619954195046206 usec\nrounds: 86671"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127721.85234744665,
            "unit": "iter/sec",
            "range": "stddev: 9.870980120363884e-7",
            "extra": "mean: 7.829513756813217 usec\nrounds: 15774"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2402.9109623199743,
            "unit": "iter/sec",
            "range": "stddev: 0.000010417415725432319",
            "extra": "mean: 416.1619034916363 usec\nrounds: 3523"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1525.5966473881383,
            "unit": "iter/sec",
            "range": "stddev: 0.000021489237573908893",
            "extra": "mean: 655.4812516873488 usec\nrounds: 1037"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12328.325448646978,
            "unit": "iter/sec",
            "range": "stddev: 0.0000039102163084849794",
            "extra": "mean: 81.1140169981276 usec\nrounds: 5883"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 908.6664047116966,
            "unit": "iter/sec",
            "range": "stddev: 0.00015157120216930655",
            "extra": "mean: 1.1005138902623808 msec\nrounds: 647"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 937.1913030180548,
            "unit": "iter/sec",
            "range": "stddev: 0.000019470555641418887",
            "extra": "mean: 1.0670180109222964 msec\nrounds: 824"
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
          "id": "65de071ed7e34d78f311c842d05e2c8dd6acd1f0",
          "message": "feat: edge-centric aggregation for untyped group nodes\n\nDiskGraph::count_edges_grouped_by_peer() does a single sequential scan\nof edge_endpoints to count edges by peer (target/source). O(E) sequential\nI/O — no random access, no per-node binary search.\n\nFusedMatchReturnAggregate uses this for 3-element patterns with untyped\ngroup nodes (e.g., MATCH (a)-[:P31]->(b) RETURN b.title, count(a)).\nPreviously OOM'd trying to iterate 124M nodes. Now does one pass over\nedge_endpoints (13 GB sequential) + HashMap accumulation + top-K heap.\n\nNote: on 16 GB RAM with external drive, the 13 GB sequential scan causes\npage cache pressure and may be killed by the OS. Works on machines with\nsufficient RAM or SSD storage.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T22:20:30+02:00",
          "tree_id": "ee5ea69d6535a809eb1ae0812509c74ccec95d54",
          "url": "https://github.com/kkollsga/kglite/commit/65de071ed7e34d78f311c842d05e2c8dd6acd1f0"
        },
        "date": 1776284577886,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1064.4351237273975,
            "unit": "iter/sec",
            "range": "stddev: 0.000019144518633019958",
            "extra": "mean: 939.465428854169 usec\nrounds: 506"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 733.2025779201366,
            "unit": "iter/sec",
            "range": "stddev: 0.000052211680352632026",
            "extra": "mean: 1.3638795472278387 msec\nrounds: 667"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12370.799035040263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060447774727889",
            "extra": "mean: 80.83552219767714 usec\nrounds: 6352"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1584.5520708534355,
            "unit": "iter/sec",
            "range": "stddev: 0.000023260280579514034",
            "extra": "mean: 631.0931766738362 usec\nrounds: 866"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 668609.0714099146,
            "unit": "iter/sec",
            "range": "stddev: 4.312835165708045e-7",
            "extra": "mean: 1.4956422859942238 usec\nrounds: 110169"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 124041.05368995387,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013596077443291096",
            "extra": "mean: 8.06184702767476 usec\nrounds: 24292"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2495.4249272154475,
            "unit": "iter/sec",
            "range": "stddev: 0.00003923443197162044",
            "extra": "mean: 400.73335370415776 usec\nrounds: 4238"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1614.0542469619904,
            "unit": "iter/sec",
            "range": "stddev: 0.00002880538390976241",
            "extra": "mean: 619.5578629914222 usec\nrounds: 1197"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13623.693281380636,
            "unit": "iter/sec",
            "range": "stddev: 0.000005230801954708945",
            "extra": "mean: 73.40153505706782 usec\nrounds: 9242"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 989.7612458666613,
            "unit": "iter/sec",
            "range": "stddev: 0.00025653338107429753",
            "extra": "mean: 1.0103446706729495 msec\nrounds: 832"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1036.937539769485,
            "unit": "iter/sec",
            "range": "stddev: 0.00003215899182816408",
            "extra": "mean: 964.3782403926699 usec\nrounds: 1015"
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
          "id": "696a7d71b650d90ca255be7d241407e5a1b3a473",
          "message": "feat: madvise sequential/dontneed for edge-centric scans\n\nAdd advise_sequential() and advise_dontneed() to MmapOrVec. Used by\ncount_edges_grouped_by_peer() to:\n1. MADV_SEQUENTIAL before scanning 13 GB edge_endpoints — enables\n   aggressive kernel readahead and reduces page cache pollution\n2. MADV_DONTNEED after scan — releases page cache pages to reduce\n   memory pressure on constrained machines\n\nThis should prevent OOM on 16 GB machines by avoiding page cache\neviction of other hot data during the sequential scan.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T23:38:42+02:00",
          "tree_id": "11358536fce3b30fd0b8c15ea920faf0a26496c1",
          "url": "https://github.com/kkollsga/kglite/commit/696a7d71b650d90ca255be7d241407e5a1b3a473"
        },
        "date": 1776289277726,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1142.2854645100583,
            "unit": "iter/sec",
            "range": "stddev: 0.00002314820667661301",
            "extra": "mean: 875.4379102853362 usec\nrounds: 457"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 752.0057578107057,
            "unit": "iter/sec",
            "range": "stddev: 0.0000351823007965987",
            "extra": "mean: 1.3297770523875685 msec\nrounds: 649"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12382.41438417497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042142750945710875",
            "extra": "mean: 80.75969427077361 usec\nrounds: 5603"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1562.2605853144285,
            "unit": "iter/sec",
            "range": "stddev: 0.000021736014227584473",
            "extra": "mean: 640.0980792834474 usec\nrounds: 782"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 648227.0616557408,
            "unit": "iter/sec",
            "range": "stddev: 3.954925407281693e-7",
            "extra": "mean: 1.5426693193674135 usec\nrounds: 98766"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127684.879337427,
            "unit": "iter/sec",
            "range": "stddev: 0.000010820630811681085",
            "extra": "mean: 7.8317809061584 usec\nrounds: 17084"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2459.4454353277856,
            "unit": "iter/sec",
            "range": "stddev: 0.000022451177821669647",
            "extra": "mean: 406.5957250508076 usec\nrounds: 4419"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1571.0167778113546,
            "unit": "iter/sec",
            "range": "stddev: 0.00003288589881596983",
            "extra": "mean: 636.5304394731796 usec\nrounds: 1140"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13457.37586316671,
            "unit": "iter/sec",
            "range": "stddev: 0.00000336308956143421",
            "extra": "mean: 74.30869213789545 usec\nrounds: 9056"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 941.2045739955927,
            "unit": "iter/sec",
            "range": "stddev: 0.000018027563271596627",
            "extra": "mean: 1.0624682748350973 msec\nrounds: 604"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 950.042336527091,
            "unit": "iter/sec",
            "range": "stddev: 0.00003404528104682045",
            "extra": "mean: 1.0525846707584956 msec\nrounds: 896"
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
          "id": "466f06a2c54adfaf7bbadaffa409fb5a2d399b18",
          "message": "feat: connection-type inverted index for unanchored edge queries\n\nBuilds a supplementary index during CSR construction mapping\nconnection_type → [source_node_ids]. Three mmap files:\n- conn_type_index_types.bin: sorted list of connection type u64s\n- conn_type_index_offsets.bin: CSR-style offset into sources array\n- conn_type_index_sources.bin: concatenated source node ID lists\n\nBuild: post-processes out_edges + edge_endpoints after merge sort.\nOne sequential pass over existing CSR arrays — no extra I/O.\n\nQuery: for untyped source nodes with typed outgoing edges\n(e.g., MATCH (a)-[:P31]->(b)), the inverted index provides exact\nsource candidates instead of iterating all 124M nodes. Binary search\non the types array + contiguous read of source IDs.\n\nExpected: MATCH (a)-[:P31]->(b) LIMIT 50 goes from TIMEOUT (cold)\nto <100ms by reading ~200 bytes from the index instead of random\npage faults across 948 MB of out_offsets.\n\nBackward compatible: older graphs without index files fall through\nto the original find_matching_nodes path.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-15T23:58:34+02:00",
          "tree_id": "1fd3be380826e8449b1e688d1b835f8b42710595",
          "url": "https://github.com/kkollsga/kglite/commit/466f06a2c54adfaf7bbadaffa409fb5a2d399b18"
        },
        "date": 1776290468647,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1051.080200060743,
            "unit": "iter/sec",
            "range": "stddev: 0.00001975381419995938",
            "extra": "mean: 951.4021860008484 usec\nrounds: 500"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 747.366502108539,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263746730001939",
            "extra": "mean: 1.338031604545706 msec\nrounds: 660"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12440.896192537779,
            "unit": "iter/sec",
            "range": "stddev: 0.000005292798723430548",
            "extra": "mean: 80.38006141388865 usec\nrounds: 6790"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1643.6964617403319,
            "unit": "iter/sec",
            "range": "stddev: 0.000021092378363426118",
            "extra": "mean: 608.3848345948305 usec\nrounds: 925"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 678985.1086929076,
            "unit": "iter/sec",
            "range": "stddev: 3.869982238542719e-7",
            "extra": "mean: 1.4727863500939922 usec\nrounds: 76953"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 123576.89390262941,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013629260588198216",
            "extra": "mean: 8.092127649590669 usec\nrounds: 21089"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2446.026205221807,
            "unit": "iter/sec",
            "range": "stddev: 0.00003785515578906309",
            "extra": "mean: 408.82636411056745 usec\nrounds: 4408"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1626.5227206864201,
            "unit": "iter/sec",
            "range": "stddev: 0.00004321256633547896",
            "extra": "mean: 614.808503614375 usec\nrounds: 1245"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13787.515343496787,
            "unit": "iter/sec",
            "range": "stddev: 0.000004572910663420148",
            "extra": "mean: 72.52938438046229 usec\nrounds: 11127"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 1019.5308927084309,
            "unit": "iter/sec",
            "range": "stddev: 0.00008125238762089323",
            "extra": "mean: 980.8432556108759 usec\nrounds: 802"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1037.7994392863002,
            "unit": "iter/sec",
            "range": "stddev: 0.000014716771360300345",
            "extra": "mean: 963.5773176826006 usec\nrounds: 1001"
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
          "id": "6fa8df6f0fbd8910fc7403584f4952932438d59d",
          "message": "release: v0.7.10 — inverted index, streaming aggregation, FusedNodeScanTopK\n\nMajor additions for disk-backed graphs:\n- Connection-type inverted index: maps edge types → source nodes, built\n  during CSR construction. Unanchored P31 LIMIT 50: 14.5s → 4.6s cold.\n- FusedNodeScanTopK: streaming ORDER BY + LIMIT for node scans without\n  materializing all rows. ORDER BY title: 23.6s → 9.0s.\n- Edge-centric aggregation: sequential scan of edge_endpoints with\n  madvise hints for page cache management.\n- Lightweight peer iteration: skips edge_endpoints for unnamed edges.\n- Streaming top-K for FusedMatchReturnAggregate.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T01:47:51+02:00",
          "tree_id": "8e4a40b806a3966a26fcabb9a382c8d28ebf509d",
          "url": "https://github.com/kkollsga/kglite/commit/6fa8df6f0fbd8910fc7403584f4952932438d59d"
        },
        "date": 1776297027034,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1129.5909307707655,
            "unit": "iter/sec",
            "range": "stddev: 0.00015418806169960984",
            "extra": "mean: 885.2762294379079 usec\nrounds: 462"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 794.8037378000522,
            "unit": "iter/sec",
            "range": "stddev: 0.00003061660403967734",
            "extra": "mean: 1.2581722410716303 msec\nrounds: 672"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13549.939870685883,
            "unit": "iter/sec",
            "range": "stddev: 0.00000344399239070526",
            "extra": "mean: 73.80106550608487 usec\nrounds: 6320"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1598.9185880062328,
            "unit": "iter/sec",
            "range": "stddev: 0.00001861068016239756",
            "extra": "mean: 625.4227122638854 usec\nrounds: 848"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 665143.6509164475,
            "unit": "iter/sec",
            "range": "stddev: 3.9841716285585037e-7",
            "extra": "mean: 1.5034346319357945 usec\nrounds: 79435"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 131051.04470582152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011424323295977204",
            "extra": "mean: 7.6306144849494535 usec\nrounds: 19151"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2474.8159921542137,
            "unit": "iter/sec",
            "range": "stddev: 0.00001117479014987506",
            "extra": "mean: 404.07044530593396 usec\nrounds: 4516"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1582.9967845263202,
            "unit": "iter/sec",
            "range": "stddev: 0.000019332897566016602",
            "extra": "mean: 631.7132225251044 usec\nrounds: 1101"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13360.868337151927,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033114261673948456",
            "extra": "mean: 74.84543480002327 usec\nrounds: 10069"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 933.3327601499739,
            "unit": "iter/sec",
            "range": "stddev: 0.000015733583404096103",
            "extra": "mean: 1.071429229420077 msec\nrounds: 741"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 951.9584940897436,
            "unit": "iter/sec",
            "range": "stddev: 0.00001959970558350372",
            "extra": "mean: 1.0504659669602439 msec\nrounds: 908"
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
          "id": "f0dbfc99cf8c39d070bf120e22bbc6063858ae66",
          "message": "fix: describe(types=['human']) performance on disk graphs\n\nThree fixes for describe() on large types (>1M nodes):\n\n1. Metadata-only properties: for types >1M nodes, list property names\n   from node_type_metadata instead of sampling nodes. Zero I/O — avoids\n   cold-cache page faults in 38 GB columns.bin. (157s → instant)\n\n2. Derive type_connectivity on load: when type_connectivity_cache is\n   empty but connection_type_metadata exists, derive triples from\n   metadata at load time. No rebuild needed for existing graphs.\n\n3. Build type_connectivity during ntriples build: compute from\n   connection_type_metadata + edge_type_counts before saving metadata.\n   New graphs get instant describe(types=[...]) on first load.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T07:25:07+02:00",
          "tree_id": "be2d5c29264c945ba9ba43e4239c823bd57f1ced",
          "url": "https://github.com/kkollsga/kglite/commit/f0dbfc99cf8c39d070bf120e22bbc6063858ae66"
        },
        "date": 1776317272723,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1051.145101019904,
            "unit": "iter/sec",
            "range": "stddev: 0.000046356954296262395",
            "extra": "mean: 951.3434434786607 usec\nrounds: 460"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 720.6643414645745,
            "unit": "iter/sec",
            "range": "stddev: 0.000042709765671593965",
            "extra": "mean: 1.3876085473685904 msec\nrounds: 665"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12357.473641971814,
            "unit": "iter/sec",
            "range": "stddev: 0.000005650455036396129",
            "extra": "mean: 80.92268929496463 usec\nrounds: 6791"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1585.2719799417916,
            "unit": "iter/sec",
            "range": "stddev: 0.000023364330367377127",
            "extra": "mean: 630.8065824999431 usec\nrounds: 800"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 684059.7894339982,
            "unit": "iter/sec",
            "range": "stddev: 4.764506182574926e-7",
            "extra": "mean: 1.4618605207410535 usec\nrounds: 68555"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 123434.43256602931,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013097706344726367",
            "extra": "mean: 8.101467145037232 usec\nrounds: 20210"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2561.3936646287493,
            "unit": "iter/sec",
            "range": "stddev: 0.00005470669198502749",
            "extra": "mean: 390.4124593612365 usec\nrounds: 3851"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1615.7194428956777,
            "unit": "iter/sec",
            "range": "stddev: 0.000050275167015227226",
            "extra": "mean: 618.9193330543878 usec\nrounds: 1195"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13830.01435898279,
            "unit": "iter/sec",
            "range": "stddev: 0.00000465938586301083",
            "extra": "mean: 72.30650482661906 usec\nrounds: 11188"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 975.7141955819264,
            "unit": "iter/sec",
            "range": "stddev: 0.00032345797100197086",
            "extra": "mean: 1.0248902850117798 msec\nrounds: 814"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 1025.879817932692,
            "unit": "iter/sec",
            "range": "stddev: 0.00001878376638038799",
            "extra": "mean: 974.773050916584 usec\nrounds: 982"
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
          "id": "c7935a563747a9b86fb91a99ebf50d6e616bc43f",
          "message": "fix: disk mode CSR build on save + from_blueprint storage param\n\nDisk mode CSR serialization:\n- save_disk() now calls ensure_disk_edges_built() before saving,\n  converting pending_edges to CSR format. Previously, edges from\n  add_connections() were lost on save/reload.\n\nfrom_blueprint storage parameter:\n- Added storage and path parameters to from_blueprint() and\n  BlueprintLoader, enabling disk/mapped mode graph construction\n  from JSON blueprints.\n\nAlso adds bench/benchmark_compatibility.py for cross-mode testing.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T08:37:42+02:00",
          "tree_id": "c443608015ca7b6e02f36668df316a4bf50caef2",
          "url": "https://github.com/kkollsga/kglite/commit/c7935a563747a9b86fb91a99ebf50d6e616bc43f"
        },
        "date": 1776321627646,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1153.1502534595172,
            "unit": "iter/sec",
            "range": "stddev: 0.000166387538487829",
            "extra": "mean: 867.1896806161576 usec\nrounds: 454"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 800.4212891226056,
            "unit": "iter/sec",
            "range": "stddev: 0.000030014200323668963",
            "extra": "mean: 1.249342082212938 msec\nrounds: 669"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13633.72832154032,
            "unit": "iter/sec",
            "range": "stddev: 0.000007810762573388446",
            "extra": "mean: 73.34750821021358 usec\nrounds: 6090"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1598.1543108644578,
            "unit": "iter/sec",
            "range": "stddev: 0.000021248677001275167",
            "extra": "mean: 625.721804960805 usec\nrounds: 887"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 637672.664837835,
            "unit": "iter/sec",
            "range": "stddev: 3.9473341241177255e-7",
            "extra": "mean: 1.568202708288127 usec\nrounds: 63436"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130364.18458889789,
            "unit": "iter/sec",
            "range": "stddev: 9.61502144160042e-7",
            "extra": "mean: 7.670818508576492 usec\nrounds: 18089"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2439.878924772418,
            "unit": "iter/sec",
            "range": "stddev: 0.000011348918799485342",
            "extra": "mean: 409.85640305626066 usec\nrounds: 4188"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1587.6490727996295,
            "unit": "iter/sec",
            "range": "stddev: 0.00002294773178623908",
            "extra": "mean: 629.8621131914368 usec\nrounds: 1175"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13378.719715097352,
            "unit": "iter/sec",
            "range": "stddev: 0.000007435382641164541",
            "extra": "mean: 74.7455676847419 usec\nrounds: 10150"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 929.7180640075277,
            "unit": "iter/sec",
            "range": "stddev: 0.000029005178736432414",
            "extra": "mean: 1.0755948913044924 msec\nrounds: 736"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 956.6260895645025,
            "unit": "iter/sec",
            "range": "stddev: 0.00002241466611799776",
            "extra": "mean: 1.045340505458348 msec\nrounds: 916"
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
          "id": "1baf62293ad25ef705f5bb25135588a9b85aef08",
          "message": "feat: defer_csr flag for disk mode — accumulate edges without intermediate CSR rebuilds\n\nAdds defer_csr flag to DiskGraph:\n- Set true in constructor (new graphs): edges from add_connections()\n  accumulate in pending_edges without triggering CSR builds.\n- Set false on load (existing graphs): CSR already built.\n- ensure_disk_edges_built() at save time does the single final build.\n\nThis is the correct infrastructure for disk mode built from DataFrame\nAPI. Small graphs work end-to-end (verified: build→save→reload→query).\nLarge graphs (legal 157K nodes) still have a separate CSR file-handle\nissue that needs investigation.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T08:47:36+02:00",
          "tree_id": "d84320c6187cf99255897b49232cdcbeecf24acf",
          "url": "https://github.com/kkollsga/kglite/commit/1baf62293ad25ef705f5bb25135588a9b85aef08"
        },
        "date": 1776322208072,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1133.2244392583789,
            "unit": "iter/sec",
            "range": "stddev: 0.000053077485895091496",
            "extra": "mean: 882.4377284471859 usec\nrounds: 464"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 790.2642874254681,
            "unit": "iter/sec",
            "range": "stddev: 0.00003111621225031433",
            "extra": "mean: 1.265399456753653 msec\nrounds: 659"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12557.794833389353,
            "unit": "iter/sec",
            "range": "stddev: 0.000021414561472511817",
            "extra": "mean: 79.63181539971853 usec\nrounds: 6013"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1584.06974975871,
            "unit": "iter/sec",
            "range": "stddev: 0.00001970315781902239",
            "extra": "mean: 631.2853333335371 usec\nrounds: 915"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 636616.3120254473,
            "unit": "iter/sec",
            "range": "stddev: 4.5060606871720393e-7",
            "extra": "mean: 1.5708048648932944 usec\nrounds: 64463"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 129398.24535256265,
            "unit": "iter/sec",
            "range": "stddev: 9.583594263091041e-7",
            "extra": "mean: 7.728080062255617 usec\nrounds: 18573"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2459.8018504469064,
            "unit": "iter/sec",
            "range": "stddev: 0.000010866941988543101",
            "extra": "mean: 406.53681101114546 usec\nrounds: 4323"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1586.1532396197185,
            "unit": "iter/sec",
            "range": "stddev: 0.00001963438111916196",
            "extra": "mean: 630.4561091712367 usec\nrounds: 1145"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13124.473292984561,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032921674014690888",
            "extra": "mean: 76.1935338414328 usec\nrounds: 10239"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 885.6279835330146,
            "unit": "iter/sec",
            "range": "stddev: 0.00001872337262204165",
            "extra": "mean: 1.1291422793696333 msec\nrounds: 698"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 894.0599934378938,
            "unit": "iter/sec",
            "range": "stddev: 0.00002030239476677283",
            "extra": "mean: 1.1184931742161275 msec\nrounds: 861"
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
          "id": "41881fd42aae84f72c2e1cfcc82e6a4e4779b3a4",
          "message": "docs: disk mode iterative updates — findings, analysis, and architecture proposal\n\nDocuments the current state of disk mode with DataFrame API:\n- Root cause of mmap file-handle conflict during CSR rebuild\n- How DuckDB, RocksDB, Neo4j, and SQLite handle live updates\n- Proposed CSR + Overflow + Periodic Compaction architecture\n- What's already built (overflow mechanism, tombstones, slot reuse)\n- What needs building (mmap lifecycle fix, compaction, incremental indices)\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T08:53:03+02:00",
          "tree_id": "01f084799377d5cacd393c48a59cde49143bbae6",
          "url": "https://github.com/kkollsga/kglite/commit/41881fd42aae84f72c2e1cfcc82e6a4e4779b3a4"
        },
        "date": 1776322548129,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1139.86842065804,
            "unit": "iter/sec",
            "range": "stddev: 0.00003090516329070518",
            "extra": "mean: 877.29424017441 usec\nrounds: 458"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 804.9134040927275,
            "unit": "iter/sec",
            "range": "stddev: 0.00003439022187779774",
            "extra": "mean: 1.2423696697251152 msec\nrounds: 654"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12983.74936979862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158986247031708",
            "extra": "mean: 77.01935485030933 usec\nrounds: 5701"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1521.6785719339707,
            "unit": "iter/sec",
            "range": "stddev: 0.00008617701240360764",
            "extra": "mean: 657.1690095688568 usec\nrounds: 836"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 640279.8062909872,
            "unit": "iter/sec",
            "range": "stddev: 4.3396028837625605e-7",
            "extra": "mean: 1.5618171777004806 usec\nrounds: 58669"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130084.76622774707,
            "unit": "iter/sec",
            "range": "stddev: 9.586317365301987e-7",
            "extra": "mean: 7.687295207566741 usec\nrounds: 16839"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2542.1530966919786,
            "unit": "iter/sec",
            "range": "stddev: 0.000010215602963668862",
            "extra": "mean: 393.3673394026771 usec\nrounds: 4119"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1564.831430302183,
            "unit": "iter/sec",
            "range": "stddev: 0.000026370166833791607",
            "extra": "mean: 639.0464689266184 usec\nrounds: 1062"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13528.140288559605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036587958439570645",
            "extra": "mean: 73.91999038076754 usec\nrounds: 7693"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 899.9864683890105,
            "unit": "iter/sec",
            "range": "stddev: 0.0000233657914410946",
            "extra": "mean: 1.1111278170548666 msec\nrounds: 645"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 918.4254784480962,
            "unit": "iter/sec",
            "range": "stddev: 0.00002197450069854681",
            "extra": "mean: 1.0888199679409416 msec\nrounds: 811"
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
          "id": "859c2592602a287dda75993ee70bf2cf24384848",
          "message": "release",
          "timestamp": "2026-04-16T08:55:00+02:00",
          "tree_id": "b21a8075643d04700120ef2ec7524914f84762a8",
          "url": "https://github.com/kkollsga/kglite/commit/859c2592602a287dda75993ee70bf2cf24384848"
        },
        "date": 1776322644662,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1126.3812779184593,
            "unit": "iter/sec",
            "range": "stddev: 0.00002073417871098983",
            "extra": "mean: 887.798847161229 usec\nrounds: 458"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 790.1397630494707,
            "unit": "iter/sec",
            "range": "stddev: 0.00003257529874868192",
            "extra": "mean: 1.2655988810645262 msec\nrounds: 639"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12329.63500444794,
            "unit": "iter/sec",
            "range": "stddev: 0.000003663077647348853",
            "extra": "mean: 81.10540171215513 usec\nrounds: 5957"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1558.915016221202,
            "unit": "iter/sec",
            "range": "stddev: 0.000018035662744074805",
            "extra": "mean: 641.471786206789 usec\nrounds: 870"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 645037.3155303857,
            "unit": "iter/sec",
            "range": "stddev: 5.093844295667833e-7",
            "extra": "mean: 1.5502979066842733 usec\nrounds: 64923"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127193.33517624217,
            "unit": "iter/sec",
            "range": "stddev: 9.921651021672173e-7",
            "extra": "mean: 7.862047163197473 usec\nrounds: 14206"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2457.9638012806795,
            "unit": "iter/sec",
            "range": "stddev: 0.000009547998191269965",
            "extra": "mean: 406.84081656490116 usec\nrounds: 4105"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1544.047771542657,
            "unit": "iter/sec",
            "range": "stddev: 0.00002111667097594087",
            "extra": "mean: 647.6483554656477 usec\nrounds: 1235"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12215.65835579786,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033989518045141757",
            "extra": "mean: 81.862145360784 usec\nrounds: 9473"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 935.8171566989583,
            "unit": "iter/sec",
            "range": "stddev: 0.000019968883641911442",
            "extra": "mean: 1.0685848115110892 msec\nrounds: 695"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 943.2147338141715,
            "unit": "iter/sec",
            "range": "stddev: 0.00008661852005563685",
            "extra": "mean: 1.060203964325494 msec\nrounds: 897"
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
          "id": "4fe19e6283a1f031924b501d6fe32de00addbe79",
          "message": "release: v0.7.12 — disk-mode correctness + Wikidata query robustness\n\nShips six inter-related fixes exposed by the API benchmark (49→51/51) and the\nWikidata disk-mode Cypher benchmark, plus Phase 3 CSR-build parallelisation.\n\nFixed:\n- Disk DataFrame/blueprint builds wrote wrong node row_ids once a second node\n  type was added: `batch_operations` pass 2 modified the DiskGraph materialisation\n  arena (cleared on next access) instead of the slot row_id. Now calls\n  `DiskGraph::update_row_id` directly.\n- `InternedKey::from_str` replaced `DefaultHasher` (per-process SipHash seed)\n  with FNV-1a so `DiskNodeSlot.node_type` resolves across processes. Breaking\n  change for disk graphs saved by earlier releases.\n- `save_disk` / `load_disk_dir` persist `parent_types`, `embeddings`, and\n  `timeseries_store` — previously omitted, causing describe() drift and\n  embedding/timeseries loss across reloads.\n- Streaming HAVING aggregate no longer OOMs on 124M-node Wikidata graphs.\n  Planner permits HAVING fusion; executor non-top-k path uses edge-centric\n  `count_edges_grouped_by_peer` and applies HAVING post-aggregation.\n- `describe()` now deterministic: `compute_join_candidates` sorts property\n  iteration and uses a stable tiebreaker.\n- `count_edges_grouped_by_peer` and pattern-matching parallel expansion now\n  honour the query deadline, so unanchored Wikidata aggregates terminate at\n  the default 20 s timeout instead of running unbounded.\n\nAdded:\n- Parallel Phase 3 CSR build: per-node sort-by-conn_type and\n  `build_conn_type_index` are now Rayon-parallelised, cutting the two new\n  serial passes from ~1000 s to ~100-200 s on Wikidata-scale builds.\n- `bench/benchmark_wikidata_cypher.py` streams CSV row-by-row so partial\n  runs (SIGKILL/OOM/Ctrl-C) still leave a populated CSV.\n- `MmapOrVec::as_mut_slice` for safe parallel writes to disjoint ranges.\n\nVerified: `make lint` clean, 1786 tests pass, `bench/api_benchmark.py` 51/51\nacross memory, mapped, and disk modes. Full Wikidata rebuild (124M nodes,\n862M edges) completes cleanly in ~77 min; cross-process reload confirmed.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T22:34:36+02:00",
          "tree_id": "c696c19fcec0aad95ee11a919d16214adae80d36",
          "url": "https://github.com/kkollsga/kglite/commit/4fe19e6283a1f031924b501d6fe32de00addbe79"
        },
        "date": 1776371879928,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1161.6629491736255,
            "unit": "iter/sec",
            "range": "stddev: 0.000029937756231303417",
            "extra": "mean: 860.8348925231471 usec\nrounds: 428"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 811.7794708935713,
            "unit": "iter/sec",
            "range": "stddev: 0.000046728021214227366",
            "extra": "mean: 1.231861651908053 msec\nrounds: 655"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12600.061929034371,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036441491404926576",
            "extra": "mean: 79.36468928741502 usec\nrounds: 5629"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1622.5221671594077,
            "unit": "iter/sec",
            "range": "stddev: 0.000025747423667213243",
            "extra": "mean: 616.3243992843109 usec\nrounds: 839"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 677204.2584485068,
            "unit": "iter/sec",
            "range": "stddev: 3.9419987544782795e-7",
            "extra": "mean: 1.4766593498555765 usec\nrounds: 81183"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 125438.44588966145,
            "unit": "iter/sec",
            "range": "stddev: 9.359186119536412e-7",
            "extra": "mean: 7.972037543255462 usec\nrounds: 17633"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2197.7725324134067,
            "unit": "iter/sec",
            "range": "stddev: 0.00007101916288739103",
            "extra": "mean: 455.0061415600117 usec\nrounds: 2423"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1637.396790171826,
            "unit": "iter/sec",
            "range": "stddev: 0.000020819740764463765",
            "extra": "mean: 610.7255162599051 usec\nrounds: 1230"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13802.733148310479,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035718045510496123",
            "extra": "mean: 72.44941920234145 usec\nrounds: 9270"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 831.3059160781136,
            "unit": "iter/sec",
            "range": "stddev: 0.00001931254567627197",
            "extra": "mean: 1.2029266009771005 msec\nrounds: 614"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 827.1397988890355,
            "unit": "iter/sec",
            "range": "stddev: 0.000018383508147357732",
            "extra": "mean: 1.208985471794659 msec\nrounds: 780"
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
          "id": "cf97b315e8b5622a25554a304b04fd3cb127ed7d",
          "message": "release: v0.7.14 — Wikidata Cypher aggregate + anchored-count perf\n\nTurns five Wikidata Cypher timeouts into sub-second queries by wiring up\ntwo new disk-mode fast paths and unblocking a cache that had been missed\nin v0.7.12.\n\nAdded:\n- Per-(conn_type, peer) edge-count histogram as a persistent disk cache.\n  Built during CSR construction via a single parallel scan of\n  edge_endpoints.bin, stored as three flat peer_count_*.bin files.\n  Replaces the 13 GB sequential scan for unanchored aggregate queries\n  (MATCH (a)-[:T]->(b) RETURN b, count(a) ORDER BY cnt DESC LIMIT N)\n  with a HashMap lookup — ms instead of tens of seconds.\n- FusedCountAnchoredEdges planner rule. Anchored counts like\n  MATCH (m)-[:P31]->({id: 5}) RETURN count(m) are fused into O(log D)\n  CSR offset arithmetic; the anchor resolves to a NodeIndex at plan\n  time via id_indices.\n- Tombstone-free short-circuit in count_edges_filtered. When no\n  removals have happened and there's no peer-type filter, returns\n  end - start directly after the binary search. Adds has_tombstones\n  to DiskGraphMeta (conservatively true for legacy graphs).\n- Bounded sources_for_conn_type. LIMIT-bounded pattern queries no\n  longer pay the 400 MB eager source copy on cold cache.\n- FusedCountTypedEdge now uses the cached edge-type counts (one-liner\n  miss from v0.7.12). 64 s scan → sub-ms lookup.\n- rebuild_caches() rebuilds the peer-count histogram on existing\n  graphs, so users can upgrade without a full rebuild.\n\nFixed:\n- DataFrame / blueprint disk builds now compact overflow into CSR at\n  save time, so conn_type_index and peer_count_histogram reflect\n  every live edge. Previously the first add_connections triggered\n  a partial CSR build; subsequent batches went to overflow and\n  never refreshed the indexes.\n- lookup_peer_counts returns None on cache miss (was Some(empty),\n  which suppressed the correct fallback to count_edges_grouped_by_peer).\n- Deadline plumbed through try_count_simple_pattern and\n  count_edges_filtered (every 1 M iterations). Closes the bypass that\n  let Q5-class hub counts run 5× past the 20 s default timeout.\n- Deadline check added to expand_var_length_fast inner edge loop.\n\nWikidata Cypher benchmark (124 M nodes, 862 M edges) on USB SSD —\n42/44 OK, zero crashes. Selected comparisons vs v0.7.12:\n\n  unanchored_P31_count:   64 s → 0.7 ms (~90 000×)\n  Q5_count_P31_incoming:  100 s TIMEOUT → 615 ms (~160×)\n  Q5_incoming_all_count:  20 s TIMEOUT → 670 ms\n  Q515_count_P31:         1.3 s → 12 ms\n  limit_10_P31 (cold):    2.6 s → 10 ms (~260×)\n  cross_type_limited:     2.5 s → 3 ms (~830×)\n  varpath_Q42_1_2:        31 s TIMEOUT → 92 ms\n\nRemaining: agg_P31_by_target (53 s), agg_P27_by_country and agg_having\n(TIMEOUT) — histogram lookup apparently misses. Tracked in\ndev-documentation/disk-mode-remaining-bugs.md.\n\nVerified: make test (1786 pass), make lint clean, api_benchmark 51/51\nacross memory, mapped, disk.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T00:24:39+02:00",
          "tree_id": "00d5f06fc3f3d1f72e06b2f17c9c30c9c37511e6",
          "url": "https://github.com/kkollsga/kglite/commit/cf97b315e8b5622a25554a304b04fd3cb127ed7d"
        },
        "date": 1776378444755,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1169.908923437785,
            "unit": "iter/sec",
            "range": "stddev: 0.000021270703785319005",
            "extra": "mean: 854.7673925432532 usec\nrounds: 456"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 819.7284514481481,
            "unit": "iter/sec",
            "range": "stddev: 0.0000323180957065391",
            "extra": "mean: 1.2199161786240074 msec\nrounds: 683"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12279.606457727066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037031438274193296",
            "extra": "mean: 81.43583456379743 usec\nrounds: 5960"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1631.5776611801896,
            "unit": "iter/sec",
            "range": "stddev: 0.000019929458637179455",
            "extra": "mean: 612.90370896391 usec\nrounds: 859"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 662715.2907168539,
            "unit": "iter/sec",
            "range": "stddev: 4.051578781420419e-7",
            "extra": "mean: 1.5089436052068572 usec\nrounds: 106907"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 129703.40084593518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010767279126455884",
            "extra": "mean: 7.709898071121698 usec\nrounds: 17885"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2120.242993945621,
            "unit": "iter/sec",
            "range": "stddev: 0.00001084813522277897",
            "extra": "mean: 471.64405346722606 usec\nrounds: 3591"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1608.9772699075063,
            "unit": "iter/sec",
            "range": "stddev: 0.000020023123243113883",
            "extra": "mean: 621.5128197911 usec\nrounds: 1243"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12085.095909793525,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038184348136626336",
            "extra": "mean: 82.74655058298873 usec\nrounds: 9005"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 811.3664238780603,
            "unit": "iter/sec",
            "range": "stddev: 0.00001966292164726437",
            "extra": "mean: 1.232488762870337 msec\nrounds: 641"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 818.2296249837308,
            "unit": "iter/sec",
            "range": "stddev: 0.00011062625382041604",
            "extra": "mean: 1.2221508112956476 msec\nrounds: 779"
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
          "id": "70d61f15403ca520ebd2f3eae72c2477be3da703",
          "message": "release: v0.7.15 — Cypher correctness fixes + per-row perf wins\n\nThree silent-correctness bugs fixed; five per-row hot paths sped up.\nInspired by cross-referencing mrmagooey/kglite; implementations here\nare independent.\n\nAdded:\n- WHERE n:Label predicate. Cypher now supports label checks as boolean\n  predicates (not just MATCH-level filters). Composes with AND/OR/NOT\n  and chained n:A:B form. Example:\n  MATCH (n) WHERE n:Person OR n:Org RETURN count(n).\n- Value::as_str() -> Option<&str> borrowing companion to as_string().\n\nFixed:\n- HAVING with aggregate expressions. HAVING count(m) > 1 was silently\n  returning zero rows when the RETURN item was aliased (count(m) AS c).\n  Root cause: the aggregate function call fell through to per-row\n  scalar dispatch, which errored, and the error was swallowed by\n  unwrap_or(false), dropping every row. evaluate_expression now\n  resolves aggregate FunctionCalls from row.projected first; a new\n  augment_rows_with_aggregate_keys helper seeds both alias and\n  expression-string keys before all three HAVING sites.\n- rand() / random() correctness under tight loops. SystemTime-per-call\n  seeding could return identical values for adjacent rows when two\n  calls resolved to the same nanosecond, and constant folding could\n  collapse rand() to a single value per query. Replaced with a\n  thread-local xorshift64 PRNG, seeded once per thread with a\n  splitmix64-avalanched counter so Rayon workers don't collide. Top\n  53 bits feed the f64 mantissa for full precision. Marked as\n  row-dependent so constant folding bypasses it.\n\nChanged (perf):\n- Function names lowercased at parse time instead of per-row during\n  dispatch. Every scalar/aggregate dispatch used to call to_lowercase()\n  on the function name each row (21+ sites); normalized once in\n  parse_function_call and compared directly. ~9-12% win on function-\n  heavy queries.\n- count(DISTINCT n) uses typed identity sets — HashSet<usize> keyed\n  on node/edge indices (with HashSet<Value> fallback for non-binding\n  expressions) instead of per-row format!(\"n:{}\", idx.index()) string\n  formatting. ~20-26% faster.\n- substring() skips intermediate Vec<char> — uses chars().skip(start)\n  .take(len).collect() instead of materializing the full char vector.\n  ~10-18% faster.\n- Zero-allocation property iterators. PropertyStorage::keys() and\n  ::iter() return explicit PropertyKeyIter / PropertyIter enums\n  instead of Box<dyn Iterator>. Saves one heap allocation per keys(n)\n  / RETURN n {.*} / property-scan call. ~12% faster on keys(n) over\n  all nodes.\n\nVerified: 1799 Python tests pass (including 13 new regression tests\nin tests/test_cypher_tier_a_fixes.py), make lint clean, perf baseline\nrecorded in tests/benchmarks/bench_tier_b.py.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T02:50:01+02:00",
          "tree_id": "ef6a4e2bb6198136f15a504d4435f7096e8c19f7",
          "url": "https://github.com/kkollsga/kglite/commit/70d61f15403ca520ebd2f3eae72c2477be3da703"
        },
        "date": 1776403314009,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1072.4085880703967,
            "unit": "iter/sec",
            "range": "stddev: 0.00002047446763877032",
            "extra": "mean: 932.480410101263 usec\nrounds: 495"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 773.7659940735552,
            "unit": "iter/sec",
            "range": "stddev: 0.000031472034681982044",
            "extra": "mean: 1.2923803936322105 msec\nrounds: 691"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12615.602212698186,
            "unit": "iter/sec",
            "range": "stddev: 0.000005029788189579452",
            "extra": "mean: 79.26692544200972 usec\nrounds: 6277"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1666.596799498,
            "unit": "iter/sec",
            "range": "stddev: 0.000034416742201188845",
            "extra": "mean: 600.0251532351512 usec\nrounds: 881"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 703512.7400854088,
            "unit": "iter/sec",
            "range": "stddev: 4.2855192616884766e-7",
            "extra": "mean: 1.421438366387788 usec\nrounds: 120251"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126739.02593744686,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013333896974485855",
            "extra": "mean: 7.890229490114265 usec\nrounds: 20807"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2099.8315895094875,
            "unit": "iter/sec",
            "range": "stddev: 0.00001836506766073706",
            "extra": "mean: 476.2286675730963 usec\nrounds: 3309"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1655.5716330254688,
            "unit": "iter/sec",
            "range": "stddev: 0.000025488499256561664",
            "extra": "mean: 604.0209798548876 usec\nrounds: 1241"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12917.21326177235,
            "unit": "iter/sec",
            "range": "stddev: 0.000005085755142654887",
            "extra": "mean: 77.41607881937158 usec\nrounds: 9858"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 838.5356170991645,
            "unit": "iter/sec",
            "range": "stddev: 0.0003967144715395387",
            "extra": "mean: 1.1925551874104126 msec\nrounds: 699"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 868.0122675212717,
            "unit": "iter/sec",
            "range": "stddev: 0.000016847077793902915",
            "extra": "mean: 1.1520574505883856 msec\nrounds: 850"
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
          "id": "aa991da6b846466e84fc385a8225d59199c2b040",
          "message": "release: v0.7.16 — CI fix (clippy 1.95) + dep bumps\n\nFollow-up to v0.7.15, which broke CI on clippy 1.95's stricter lints.\nNo functional changes to the Cypher engine or storage; all changes are\neither mechanical lint fixes or major-version dep bumps with absorbed\nAPI changes.\n\nDependencies:\n- pyo3 0.27 → 0.28: added #[pyclass(skip_from_py_object)] on\n  KnowledgeGraph (pyo3 0.28 made the FromPyObject derive opt-in).\n- geo 0.29 → 0.33: Geodesic is now a static value (Geodesic.distance(...),\n  .length(&Geodesic)); LengthMeasurable trait imported from\n  geo::line_measures.\n- wkt 0.11 → 0.14, bzip2 0.5 → 0.6: API-compatible.\n\nClippy 1.95 compat (30 lint fixes):\n- sort_by(|a, b| b.X.cmp(&a.X)) → sort_by_key(|x| Reverse(x.X))\n  (11 sites across introspection.rs, ntriples.rs, schema.rs).\n- Collapsed `if` into outer match arm guards (7 sites in\n  column_store.rs, mmap_column_store.rs, cypher/executor.rs,\n  filtering_methods.rs, schema.rs, temporal.rs — temporal.rs reworked\n  to nested if-let instead since edition 2021 doesn't support let chains).\n- file_len.checked_div(elem_size).unwrap_or(len) in mmap_vec.rs.\n- Removed redundant .into_iter() on IntoIterator args (4 sites in\n  io_operations.rs, schema.rs).\n\nStubtest allowlist: removed unused entries for collect_children /\ntraverse (the ... vs None mismatch no longer fires).\n\nVerified: make test (1799 pass), make lint clean, 107 spatial tests\npass after geo/wkt upgrade.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T07:43:17+02:00",
          "tree_id": "9b857962d9218adc7fc955a2611fd1991b9b178e",
          "url": "https://github.com/kkollsga/kglite/commit/aa991da6b846466e84fc385a8225d59199c2b040"
        },
        "date": 1776404841290,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1056.8399394414807,
            "unit": "iter/sec",
            "range": "stddev: 0.000021829328534489697",
            "extra": "mean: 946.2170785563617 usec\nrounds: 471"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 763.5194436477532,
            "unit": "iter/sec",
            "range": "stddev: 0.00003361434445166338",
            "extra": "mean: 1.3097243407744128 msec\nrounds: 672"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12624.63467066918,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052502360976309245",
            "extra": "mean: 79.21021289616408 usec\nrounds: 6839"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1692.5044140057214,
            "unit": "iter/sec",
            "range": "stddev: 0.00008377144135835585",
            "extra": "mean: 590.8404088786143 usec\nrounds: 856"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 674141.6097485508,
            "unit": "iter/sec",
            "range": "stddev: 5.394905784726164e-7",
            "extra": "mean: 1.4833678644654371 usec\nrounds: 92670"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128438.1326272956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015479682134623405",
            "extra": "mean: 7.785849728147483 usec\nrounds: 20769"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2195.5636473831923,
            "unit": "iter/sec",
            "range": "stddev: 0.000019257244364771408",
            "extra": "mean: 455.4639084099709 usec\nrounds: 3603"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1686.5994066865755,
            "unit": "iter/sec",
            "range": "stddev: 0.000045186511989245356",
            "extra": "mean: 592.9090191989094 usec\nrounds: 1198"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13272.6766327152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052439583007070015",
            "extra": "mean: 75.34275321189901 usec\nrounds: 9340"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 862.1324551604122,
            "unit": "iter/sec",
            "range": "stddev: 0.00008761141809310718",
            "extra": "mean: 1.159914574627556 msec\nrounds: 670"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 888.7865569360266,
            "unit": "iter/sec",
            "range": "stddev: 0.000016408385799134443",
            "extra": "mean: 1.125129528789642 msec\nrounds: 851"
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
          "id": "fa63011c4b1747a0421bdbddfa69df538db95f76",
          "message": "release: v0.7.17 — Python 3.14 wheels\n\nAdds Python 3.14 to the CI test matrix and wheel-build pipeline across\nLinux, macOS (Intel + arm64), and Windows. Enabled by pyo3 0.28 (bumped\nin 0.7.16), which supports up to ABI3_MAX_MINOR = 14.\n\nVerified locally: full test suite (1758 passed, 4 skipped — matching\n3.12 once optional tree-sitter code-tree tests are excluded) against\nCPython 3.14.3 on macOS aarch64, release build.\n\nCI changes:\n- .github/workflows/ci.yml: add 3.14 to python-version matrix\n- .github/workflows/build_wheels.yml: add 3.14 include entries for all\n  four OS/target combos\n- pyproject.toml: add 3.14 classifier\n- .gitignore: cover .venv* (added during the 3.14 compat test)\n\nNo functional changes to kglite itself.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T07:51:59+02:00",
          "tree_id": "4147e4c6149584babdc4754d8687f79e8c1e9a68",
          "url": "https://github.com/kkollsga/kglite/commit/fa63011c4b1747a0421bdbddfa69df538db95f76"
        },
        "date": 1776406439641,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1065.353349328723,
            "unit": "iter/sec",
            "range": "stddev: 0.00002323997714455524",
            "extra": "mean: 938.6557057619408 usec\nrounds: 486"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 762.7324550210269,
            "unit": "iter/sec",
            "range": "stddev: 0.00012211546551123745",
            "extra": "mean: 1.3110757165465474 msec\nrounds: 695"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13651.233611057052,
            "unit": "iter/sec",
            "range": "stddev: 0.000004692993442782003",
            "extra": "mean: 73.25345302054113 usec\nrounds: 7333"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1744.7701000895584,
            "unit": "iter/sec",
            "range": "stddev: 0.00002029508839631956",
            "extra": "mean: 573.1414126988252 usec\nrounds: 945"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 701277.1211722958,
            "unit": "iter/sec",
            "range": "stddev: 4.29986577618281e-7",
            "extra": "mean: 1.4259698053864092 usec\nrounds: 95878"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 127078.32107406987,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013710956607352962",
            "extra": "mean: 7.869162824531905 usec\nrounds: 18609"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2259.824741838408,
            "unit": "iter/sec",
            "range": "stddev: 0.000014454559682153928",
            "extra": "mean: 442.51219197931346 usec\nrounds: 3516"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1699.1127785378312,
            "unit": "iter/sec",
            "range": "stddev: 0.000040187892125741845",
            "extra": "mean: 588.5424514672584 usec\nrounds: 1329"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13892.496026241244,
            "unit": "iter/sec",
            "range": "stddev: 0.00000464604193783822",
            "extra": "mean: 71.98130545519832 usec\nrounds: 11347"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 876.1129886150237,
            "unit": "iter/sec",
            "range": "stddev: 0.000026242477287154454",
            "extra": "mean: 1.1414052901793172 msec\nrounds: 672"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 898.7609742934139,
            "unit": "iter/sec",
            "range": "stddev: 0.000017600329166331525",
            "extra": "mean: 1.1126428812579205 msec\nrounds: 859"
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
          "id": "c5a05aac1036c349552cdffe94a366d7e6eb4a94",
          "message": "docs: Phase 1 report-out in todo.md\n\nCloses Phase 1. Records what shipped (7 commits on main), benchmark\ndeltas vs Phase 0, surprises found during investigation (grep gate\nalmost already met; connection_type_metadata lives on DirGraph;\nedge_references/edges/edge_weight still inherent-only), design\ndecisions (enum-wrapped iterators not GATs, node_data escape hatch,\ndisk-only trait helpers keep Option/fallback contract, MappedGraph\nstays aliased), debt carried forward (inherent methods still on\nGraphBackend for non-Phase-1 callers; trait missing edge_references\nand friends; consumers not reshaped to &impl GraphRead yet), and\nPhase 2 prerequisites (GraphWrite trait; mutation surface; OCC\nsemantics decision).\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T13:53:00+02:00",
          "tree_id": "788b0f664463b9795a9ff01aa3379ffe5f6abd4a",
          "url": "https://github.com/kkollsga/kglite/commit/c5a05aac1036c349552cdffe94a366d7e6eb4a94"
        },
        "date": 1776427112042,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1126.6652303941305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000998504881442839",
            "extra": "mean: 887.5750959760953 usec\nrounds: 323"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 746.801336221305,
            "unit": "iter/sec",
            "range": "stddev: 0.0003070192313239983",
            "extra": "mean: 1.3390442029199354 msec\nrounds: 685"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12532.172789247332,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037355486115081",
            "extra": "mean: 79.79462275352644 usec\nrounds: 5731"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1640.4220504932816,
            "unit": "iter/sec",
            "range": "stddev: 0.00014528297975490706",
            "extra": "mean: 609.5992185055644 usec\nrounds: 897"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 667251.5886848484,
            "unit": "iter/sec",
            "range": "stddev: 4.417407837387475e-7",
            "extra": "mean: 1.4986850791483286 usec\nrounds: 70767"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 133491.83737453198,
            "unit": "iter/sec",
            "range": "stddev: 9.870967323322663e-7",
            "extra": "mean: 7.49109473408734 usec\nrounds: 14926"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2304.890461704045,
            "unit": "iter/sec",
            "range": "stddev: 0.000011242004776263684",
            "extra": "mean: 433.86009730834803 usec\nrounds: 3566"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1638.614099533585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000195454968775798",
            "extra": "mean: 610.2718146295946 usec\nrounds: 1203"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13709.517752571568,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032943126072650837",
            "extra": "mean: 72.94202597406641 usec\nrounds: 9856"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 825.668443880499,
            "unit": "iter/sec",
            "range": "stddev: 0.000019628750505958943",
            "extra": "mean: 1.2111399041728816 msec\nrounds: 647"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 845.0652910765692,
            "unit": "iter/sec",
            "range": "stddev: 0.00001988227763278138",
            "extra": "mean: 1.1833405188444692 msec\nrounds: 796"
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
          "id": "9f491f87efccfe90ad778a7d808e70b727ff8d74",
          "message": "refactor: Phase 2 — GraphWrite trait + mutation-path migration\n\nIntroduces trait GraphWrite: GraphRead (7 methods) in\nsrc/graph/storage/mod.rs. Write-path callers in batch_operations.rs,\ncypher/executor.rs (CREATE / SET / DELETE paths), maintain_graph.rs,\nand subgraph.rs now route through UFCS trait dispatch. The one\nGraphBackend::Disk(ref mut dg) enum match in batch_operations.rs\ncollapses into a default-no-op GraphWrite::update_row_id override.\n\nTransaction boundary decision: transactions stay on DirGraph (OCC\nversion, read_only, schema_locked — all DirGraph-level). No\nGraphTransaction trait. Documented in ARCHITECTURE.md.\n\nNew parity suite tests/test_phase2_parity.py (14 tests, @parity):\nconflict matrix, mid-batch failure, schema-locked rejection, tombstone\nvisibility, MERGE idempotency, collect-then-delete snapshot. Three\nxfail=strict tests pin pre-existing disk-mode mutation bugs for\nPhase 5: add_nodes(conflict='update') and ='replace' on disk don't\napply property changes; MERGE edge isn't visible after creation on\ndisk.\n\nBenchmarks hold vs Phase 1 (in-memory within ±5%, mapped/disk within\ngates). 477 Rust tests + 1799 Python tests + 34 parity tests green.\n\nReport-out appended to todo.md Phase 2 section.",
          "timestamp": "2026-04-17T14:23:25+02:00",
          "tree_id": "96c0c4df770cd5d8339f741e3bca1439244dcbf9",
          "url": "https://github.com/kkollsga/kglite/commit/9f491f87efccfe90ad778a7d808e70b727ff8d74"
        },
        "date": 1776428797511,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1070.6807435192352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000459307281858542",
            "extra": "mean: 933.9852295401207 usec\nrounds: 501"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 775.3591867753436,
            "unit": "iter/sec",
            "range": "stddev: 0.00002803986152340148",
            "extra": "mean: 1.2897248359936504 msec\nrounds: 689"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12605.595517404841,
            "unit": "iter/sec",
            "range": "stddev: 0.000005366400032983406",
            "extra": "mean: 79.32984987653114 usec\nrounds: 6095"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1693.8673748408835,
            "unit": "iter/sec",
            "range": "stddev: 0.00002026494129318572",
            "extra": "mean: 590.3649924740636 usec\nrounds: 930"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 695930.1193488062,
            "unit": "iter/sec",
            "range": "stddev: 4.201977186066572e-7",
            "extra": "mean: 1.4369258812015742 usec\nrounds: 86874"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 125992.80154551053,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013162206356220515",
            "extra": "mean: 7.936961379803787 usec\nrounds: 20611"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2178.465789829201,
            "unit": "iter/sec",
            "range": "stddev: 0.00001579983781751111",
            "extra": "mean: 459.0386521876037 usec\nrounds: 3246"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1694.169735199598,
            "unit": "iter/sec",
            "range": "stddev: 0.00009932738686628892",
            "extra": "mean: 590.2596293766193 usec\nrounds: 1314"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13968.761235324457,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046017276277102565",
            "extra": "mean: 71.58830931057666 usec\nrounds: 10837"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 863.6229625962583,
            "unit": "iter/sec",
            "range": "stddev: 0.0003956282490101136",
            "extra": "mean: 1.157912704166364 msec\nrounds: 720"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 896.6482949730016,
            "unit": "iter/sec",
            "range": "stddev: 0.00001827853159825575",
            "extra": "mean: 1.115264486205386 msec\nrounds: 870"
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
          "id": "a7a8cdd02e1348c538a22edb569545dbfb512013",
          "message": "refactor: Phase 3 — GATs on GraphRead + edge-method migration\n\nAdds generic associated types (NodeIndicesIter, EdgesIter,\nNeighborsIter, etc.) to every iterator-returning method on GraphRead.\nPromotes the last 7 inherent edge methods (edges, edge_references,\nedge_weight, edge_indices, find_edge, edges_connecting, edge_weights)\nonto the trait and deletes their GraphBackend inherent counterparts.\nMigrates 11 caller files off direct `graph.graph.X()` / `self.graph.graph.X()`\nsyntax for the full iteration/edge-method surface. Drops `&dyn GraphRead`\nsupport (GATs make the trait non-object-safe); one caller in\nintrospection.rs flipped to `&impl GraphRead`.\n\nAuthors tests/test_phase3_parity.py (13 parity tests) covering\ntraversal-under-mutation isolation (CREATE + DELETE variants) and\nrow-count parity across memory/mapped/disk for 3 Cypher patterns that\nexercise the migrated edge methods.\n\nPer-backend `impl GraphRead for MemoryGraph / DiskGraph` deferred to\nPhase 5 (paired with columnar cleanup + MappedGraph struct promotion).\nToday's GATs are the trait shape; Phase 5 lands the inlining payoff.\n\nIn-memory benchmarks (TestStorageModeMatrix at N=10000): two_hop -27%,\nmulti_predicate -11%, construction -28%; pagerank +4% and find +5%\nwithin single-run noise band. Mapped/disk within ±10%. make test\n+ make lint + pytest -m parity all green.\n\nSee todo.md Phase 3 Report-out for full details.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T15:14:27+02:00",
          "tree_id": "36f3c7250fba4803b3286f63e34d94c4ab54b561",
          "url": "https://github.com/kkollsga/kglite/commit/a7a8cdd02e1348c538a22edb569545dbfb512013"
        },
        "date": 1776431958263,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1166.940753144761,
            "unit": "iter/sec",
            "range": "stddev: 0.000024493349394241354",
            "extra": "mean: 856.941534782399 usec\nrounds: 460"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 822.9727563601355,
            "unit": "iter/sec",
            "range": "stddev: 0.00003460547240990663",
            "extra": "mean: 1.215107052173665 msec\nrounds: 690"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12450.826578698363,
            "unit": "iter/sec",
            "range": "stddev: 0.000005625676987818989",
            "extra": "mean: 80.31595281480037 usec\nrounds: 5489"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1639.2496619662893,
            "unit": "iter/sec",
            "range": "stddev: 0.000020255190323188828",
            "extra": "mean: 610.0352028137644 usec\nrounds: 853"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 671822.1490427906,
            "unit": "iter/sec",
            "range": "stddev: 3.798675203732899e-7",
            "extra": "mean: 1.4884891803951326 usec\nrounds: 82720"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128063.19384154967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010474876928533814",
            "extra": "mean: 7.808644857298204 usec\nrounds: 18142"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2176.3495879226375,
            "unit": "iter/sec",
            "range": "stddev: 0.00003449214264965045",
            "extra": "mean: 459.4850044080083 usec\nrounds: 3176"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1609.2995879899884,
            "unit": "iter/sec",
            "range": "stddev: 0.00008869836089616871",
            "extra": "mean: 621.388340283488 usec\nrounds: 1199"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13648.599749373983,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033971333357426395",
            "extra": "mean: 73.2675892298671 usec\nrounds: 10232"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 834.2313087207949,
            "unit": "iter/sec",
            "range": "stddev: 0.000019110754465889318",
            "extra": "mean: 1.1987083073319242 msec\nrounds: 641"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 828.4010465035476,
            "unit": "iter/sec",
            "range": "stddev: 0.00001855081346063828",
            "extra": "mean: 1.2071447811669531 msec\nrounds: 754"
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
          "id": "75c384187b421dbea3fa0bdce9d6d03cf0e36253",
          "message": "file restructuring",
          "timestamp": "2026-04-18T10:09:58+02:00",
          "tree_id": "bc164223df392ba560a667282188b3607b4f6750",
          "url": "https://github.com/kkollsga/kglite/commit/75c384187b421dbea3fa0bdce9d6d03cf0e36253"
        },
        "date": 1776499962601,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1050.1061849181008,
            "unit": "iter/sec",
            "range": "stddev: 0.000025902954619810196",
            "extra": "mean: 952.284649269056 usec\nrounds: 479"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 754.6052958609923,
            "unit": "iter/sec",
            "range": "stddev: 0.00004382559515968199",
            "extra": "mean: 1.3251961064744666 msec\nrounds: 695"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 11864.954490371683,
            "unit": "iter/sec",
            "range": "stddev: 0.00002126331896472295",
            "extra": "mean: 84.28182348373035 usec\nrounds: 6430"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1615.6012075239832,
            "unit": "iter/sec",
            "range": "stddev: 0.00004420873540611417",
            "extra": "mean: 618.9646277453375 usec\nrounds: 865"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 685661.8515986757,
            "unit": "iter/sec",
            "range": "stddev: 4.4130130391646024e-7",
            "extra": "mean: 1.4584448553881477 usec\nrounds: 73552"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 125692.31868629841,
            "unit": "iter/sec",
            "range": "stddev: 0.000001442534926715612",
            "extra": "mean: 7.955935656623453 usec\nrounds: 16614"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2017.4564140289954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000485347809937975",
            "extra": "mean: 495.6736577039269 usec\nrounds: 3310"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1628.0913384303944,
            "unit": "iter/sec",
            "range": "stddev: 0.000025438492909917133",
            "extra": "mean: 614.216153845568 usec\nrounds: 1196"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13504.336840835973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052325100584233125",
            "extra": "mean: 74.0502856072195 usec\nrounds: 10700"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 830.5408110137688,
            "unit": "iter/sec",
            "range": "stddev: 0.00018061020421707833",
            "extra": "mean: 1.2040347527045507 msec\nrounds: 647"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 861.7178099572698,
            "unit": "iter/sec",
            "range": "stddev: 0.000023236797183814423",
            "extra": "mean: 1.160472707474373 msec\nrounds: 776"
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
          "id": "f857473d80fe70b5eb4ce4cddd154cef5f17b6db",
          "message": "clean up and regression fixes",
          "timestamp": "2026-04-18T10:37:29+02:00",
          "tree_id": "6827d2d3449181154140582be414b530669336b7",
          "url": "https://github.com/kkollsga/kglite/commit/f857473d80fe70b5eb4ce4cddd154cef5f17b6db"
        },
        "date": 1776501617817,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1036.7904981295005,
            "unit": "iter/sec",
            "range": "stddev: 0.00002656693984636249",
            "extra": "mean: 964.5150122460851 usec\nrounds: 490"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 764.2805843743034,
            "unit": "iter/sec",
            "range": "stddev: 0.00003195015140138076",
            "extra": "mean: 1.3084199971122827 msec\nrounds: 692"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12425.210210889356,
            "unit": "iter/sec",
            "range": "stddev: 0.000008797294491222882",
            "extra": "mean: 80.48153576698509 usec\nrounds: 6822"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1651.556160346805,
            "unit": "iter/sec",
            "range": "stddev: 0.00003083240634239119",
            "extra": "mean: 605.4895522232881 usec\nrounds: 900"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 699088.2182692643,
            "unit": "iter/sec",
            "range": "stddev: 4.5999264302606716e-7",
            "extra": "mean: 1.4304346345239578 usec\nrounds: 76600"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 121970.9719777324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013508284881961648",
            "extra": "mean: 8.198672059304116 usec\nrounds: 18668"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2090.003135126123,
            "unit": "iter/sec",
            "range": "stddev: 0.000016047410401969245",
            "extra": "mean: 478.46818179038473 usec\nrounds: 3229"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1646.6322702290922,
            "unit": "iter/sec",
            "range": "stddev: 0.000023112058825404568",
            "extra": "mean: 607.3001349966694 usec\nrounds: 1163"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12968.09854711143,
            "unit": "iter/sec",
            "range": "stddev: 0.000005445904732442777",
            "extra": "mean: 77.11230727983204 usec\nrounds: 9148"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 847.6966962096396,
            "unit": "iter/sec",
            "range": "stddev: 0.00023957885493385708",
            "extra": "mean: 1.179667214077115 msec\nrounds: 682"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 872.8018091902187,
            "unit": "iter/sec",
            "range": "stddev: 0.00004104127994013235",
            "extra": "mean: 1.1457354802320991 msec\nrounds: 860"
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
          "id": "8763ef392790192983b311a94aa9b4ba822525cd",
          "message": "chore: release 0.8.0\n\nInternal storage-architecture refactor plus bug fixes and perf wins.\nNo Python API signature changes — kglite/__init__.pyi is byte-identical\nto v0.7.17 (diff shows docstring additions only).\n\nHighlights:\n- Wikidata-scale `save()` unblocked (60+ min → 5.5s on 124M nodes).\n- `rebuild_caches` on 862M-edge disk graphs: 235s → 169s (−28 %).\n- `load_ntriples` on Wikidata: 4747s → 4627s (−2.5 %).\n- Cypher primitives: pattern_match −60 %, two_hop −24 %, describe −21 %.\n- Concurrent `load_ntriples` no longer deletes each other's spill dirs.\n- src/graph/ reorganised into clean domain subdirectories; every file\n  ≤2,500 lines; per-backend `impl GraphRead/GraphWrite`; v3 `.kgl`\n  format deterministic on save.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-18T14:27:29+02:00",
          "tree_id": "f7979b00ea9238b3d33c05604470f5a15a33575f",
          "url": "https://github.com/kkollsga/kglite/commit/8763ef392790192983b311a94aa9b4ba822525cd"
        },
        "date": 1776515464753,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1057.792816091208,
            "unit": "iter/sec",
            "range": "stddev: 0.000023154274909960955",
            "extra": "mean: 945.3647111116089 usec\nrounds: 495"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 762.2168516008187,
            "unit": "iter/sec",
            "range": "stddev: 0.000032705589604819814",
            "extra": "mean: 1.3119625968643773 msec\nrounds: 702"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12540.530981915354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000548330196455059",
            "extra": "mean: 79.74144009070235 usec\nrounds: 7019"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1654.2333929638266,
            "unit": "iter/sec",
            "range": "stddev: 0.00003704865217551439",
            "extra": "mean: 604.5096201379046 usec\nrounds: 874"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 704490.6704551901,
            "unit": "iter/sec",
            "range": "stddev: 4.061053641779515e-7",
            "extra": "mean: 1.4194652135760344 usec\nrounds: 73175"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 123826.67266660629,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014164328421882607",
            "extra": "mean: 8.075804497246102 usec\nrounds: 13877"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2140.5917588753255,
            "unit": "iter/sec",
            "range": "stddev: 0.000015339446041100427",
            "extra": "mean: 467.1605390676658 usec\nrounds: 3430"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1629.1574442248857,
            "unit": "iter/sec",
            "range": "stddev: 0.000023231232406863894",
            "extra": "mean: 613.8142163882609 usec\nrounds: 1257"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 12941.972811146266,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067514712305924",
            "extra": "mean: 77.26797255660672 usec\nrounds: 10166"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 850.3904630363572,
            "unit": "iter/sec",
            "range": "stddev: 0.0001567808981243643",
            "extra": "mean: 1.1759304031108901 msec\nrounds: 707"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 873.3119581148949,
            "unit": "iter/sec",
            "range": "stddev: 0.0000173514840139938",
            "extra": "mean: 1.1450661939389564 msec\nrounds: 825"
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
          "id": "35cec09091b6db319c0df3639be00fa3ce7be059",
          "message": "feat: port code_tree to Rust (0.8.0 → 0.8.1)\n\nRewrite the polyglot codebase parser (previously ~7,500 LOC of Python\nunder kglite/code_tree/) as a first-class Rust module at src/code_tree/,\nexposed via PyO3 as kglite._kglite_code_tree. All eight language parsers\n(Python, Rust, TypeScript/JavaScript, Go, Java, C#, C, C++), the\n5-tier CALLS resolver, type-edge builder, manifest readers, and repo\nclone helper now run natively. Tree-sitter grammars are bundled into\nthe native extension, so `pip install kglite[code-tree]` is no longer\nrequired and the extras entry has been dropped.\n\nPerformance (release build, wall-clock):\n\n- duckdb C++ (2,805 files)  : 29 s  (Python) → 0.63 s — ~46×\n- neo4j  Java (7,966 files) : crashed       → 1.66 s\n- KGLite mixed  (248 files) : —             → 0.17 s\n\nKey optimizations: per-thread tree-sitter Parser via `thread_local!`\n(no Mutex contention), rayon-parallel 5-tier CALLS match loop,\nAho-Corasick multi-pattern scan for USES_TYPE (replacing regex\nalternation), `&str` borrows through the hot path (no String clones\nper edge).\n\nAlso:\n\n- Switch to PyO3 abi3-py310. CI wheel matrix collapses from 20 wheels\n  (5 Python versions × 4 platforms) to 4 — one wheel per platform,\n  works on Python ≥ 3.10.\n- Schema-aware edge routing (per-row node-type lookup) fixes the\n  pre-existing Python crash on pure-Java repos:\n  \"Source type 'Struct' does not exist in graph\".\n- Fix abi3 compile failure in type_conversions.rs by routing datetime\n  field access through Python's generic attribute API instead of the\n  non-stable `PyDateAccess` trait.\n- Rewrite tests/test_code_tree_{calls,pyi}.py to drive the public\n  build() API (the old tests imported the now-deleted private helpers\n  `_build_call_edges` and `PythonParser`).\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T02:06:09+02:00",
          "tree_id": "6ceb2272d6e54637343b0e74ef717b0b8975d047",
          "url": "https://github.com/kkollsga/kglite/commit/35cec09091b6db319c0df3639be00fa3ce7be059"
        },
        "date": 1776557381552,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1058.9392255257915,
            "unit": "iter/sec",
            "range": "stddev: 0.000025389581747112058",
            "extra": "mean: 944.3412576424993 usec\nrounds: 458"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 756.7316547126691,
            "unit": "iter/sec",
            "range": "stddev: 0.000032915075864795734",
            "extra": "mean: 1.321472405405982 msec\nrounds: 666"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12527.30815659611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051142693690519626",
            "extra": "mean: 79.82560878200012 usec\nrounds: 6536"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1580.8388013095891,
            "unit": "iter/sec",
            "range": "stddev: 0.00014401748460050553",
            "extra": "mean: 632.5755663206053 usec\nrounds: 867"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 668356.6607040226,
            "unit": "iter/sec",
            "range": "stddev: 4.747189849644502e-7",
            "extra": "mean: 1.4962071283117555 usec\nrounds: 71603"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 124719.4462932201,
            "unit": "iter/sec",
            "range": "stddev: 0.00000147535917010363",
            "extra": "mean: 8.01799582760304 usec\nrounds: 17257"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2226.8375483204154,
            "unit": "iter/sec",
            "range": "stddev: 0.000028464256586374266",
            "extra": "mean: 449.06733351709767 usec\nrounds: 3628"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1669.4569727480625,
            "unit": "iter/sec",
            "range": "stddev: 0.00002224395387113064",
            "extra": "mean: 598.9971687344049 usec\nrounds: 1209"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13735.353188887811,
            "unit": "iter/sec",
            "range": "stddev: 0.000004259413950263256",
            "extra": "mean: 72.80482607531498 usec\nrounds: 11551"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 870.0832501353158,
            "unit": "iter/sec",
            "range": "stddev: 0.0002775717978494181",
            "extra": "mean: 1.1493153095919035 msec\nrounds: 688"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 884.5093492375325,
            "unit": "iter/sec",
            "range": "stddev: 0.000015641062634958257",
            "extra": "mean: 1.1305702996378988 msec\nrounds: 831"
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
          "id": "08ed423641df1c16809e08f36f72ce7ebefc4e19",
          "message": "chore: release 0.8.2\n\nPromotes the blueprint Rust port + parallelisation wins from the two\npreceding commits to a tagged release. Also retroactively splits the\ncode_tree / abi3 / parallel-parsing work out of [Unreleased] into a\n[0.8.1] section — that work shipped as 0.8.1 but the changelog\nwasn't promoted at the time.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T09:49:11+02:00",
          "tree_id": "c7033d1effe0f6d786fb4b4b868f8f0376738565",
          "url": "https://github.com/kkollsga/kglite/commit/08ed423641df1c16809e08f36f72ce7ebefc4e19"
        },
        "date": 1776587470412,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1066.3339842207956,
            "unit": "iter/sec",
            "range": "stddev: 0.000018272095417186566",
            "extra": "mean: 937.7924879049334 usec\nrounds: 537"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 754.3020123664683,
            "unit": "iter/sec",
            "range": "stddev: 0.0001449330336227593",
            "extra": "mean: 1.325728930329517 msec\nrounds: 689"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13876.17617156756,
            "unit": "iter/sec",
            "range": "stddev: 0.000004931176198048278",
            "extra": "mean: 72.06596310365467 usec\nrounds: 7399"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1688.930176961134,
            "unit": "iter/sec",
            "range": "stddev: 0.0000706520756258951",
            "extra": "mean: 592.0907883825514 usec\nrounds: 964"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 687970.0291605095,
            "unit": "iter/sec",
            "range": "stddev: 6.185070491143772e-7",
            "extra": "mean: 1.4535516920994986 usec\nrounds: 46883"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 123600.69587421912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018434569102484038",
            "extra": "mean: 8.09056933641894 usec\nrounds: 25535"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2205.343904010594,
            "unit": "iter/sec",
            "range": "stddev: 0.000014503301478460074",
            "extra": "mean: 453.4440175890119 usec\nrounds: 3639"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1691.0420029552447,
            "unit": "iter/sec",
            "range": "stddev: 0.00002319764546751684",
            "extra": "mean: 591.3513669396809 usec\nrounds: 1240"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13823.551055132542,
            "unit": "iter/sec",
            "range": "stddev: 0.000004647424708817261",
            "extra": "mean: 72.3403122693796 usec\nrounds: 11410"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 815.2230409025568,
            "unit": "iter/sec",
            "range": "stddev: 0.00005249022693043784",
            "extra": "mean: 1.2266581657123814 msec\nrounds: 700"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 882.1850264944662,
            "unit": "iter/sec",
            "range": "stddev: 0.00003341313180630608",
            "extra": "mean: 1.133549051465648 msec\nrounds: 855"
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
          "id": "111e33eb1bc8d4169749217b20edb4f97323e19f",
          "message": "perf(cypher): spatial-join operator for contains() cross-product (0.8.3)\n\nPlanner pass `fuse_spatial_join` rewrites `MATCH (s:A), (w:B) WHERE\ncontains(s, w) [AND rest]` into a new `Clause::SpatialJoin`. Executor\nbuilds an R-tree (new `rstar` dep) over the container side and probes\nthe point side once, replacing the N×M cartesian with O((N+M) log N + K).\n\nSpeedups on tests/bench_spatial.py (release build):\n- contains 500K pairs: 86.96 ms → 0.52 ms (~167×)\n- contains 2.6M prospect_shape (263 × 10K complex): 480.51 ms → 3.32 ms (~145×)\n- contains 100K pairs: 17.65 ms → 0.55 ms (~32×)\n\nFires only when both types have SpatialConfig (container.geometry,\nprobe.location), patterns are disjoint typed nodes with no edges, and\nWHERE is `contains(var, var)` with at most an ANDed residual. NOT,\nconstant-point, edges, 3+ patterns, and disjunctions fall back to the\nexisting per-row fast path unchanged.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T11:25:27+02:00",
          "tree_id": "af74824a35988060c916cc3c9668d02316dd634c",
          "url": "https://github.com/kkollsga/kglite/commit/111e33eb1bc8d4169749217b20edb4f97323e19f"
        },
        "date": 1776591045387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1055.5799102972348,
            "unit": "iter/sec",
            "range": "stddev: 0.000018521288156370992",
            "extra": "mean: 947.3465630076416 usec\nrounds: 492"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 738.8663342095434,
            "unit": "iter/sec",
            "range": "stddev: 0.00003559378821794251",
            "extra": "mean: 1.353424772113651 msec\nrounds: 667"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12713.876886666609,
            "unit": "iter/sec",
            "range": "stddev: 0.00000515285429031728",
            "extra": "mean: 78.65421451805369 usec\nrounds: 6764"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1677.5281720060698,
            "unit": "iter/sec",
            "range": "stddev: 0.000022163180683862214",
            "extra": "mean: 596.1151751056148 usec\nrounds: 948"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 675651.1619820372,
            "unit": "iter/sec",
            "range": "stddev: 4.794643002716235e-7",
            "extra": "mean: 1.4800536967426778 usec\nrounds: 68980"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 124326.75670574239,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013652147943624701",
            "extra": "mean: 8.043320894848147 usec\nrounds: 18598"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2264.017348864364,
            "unit": "iter/sec",
            "range": "stddev: 0.00004214285356397561",
            "extra": "mean: 441.69272841553186 usec\nrounds: 3579"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1680.2658095487748,
            "unit": "iter/sec",
            "range": "stddev: 0.000022061006929130027",
            "extra": "mean: 595.1439315833867 usec\nrounds: 1257"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13745.024617060264,
            "unit": "iter/sec",
            "range": "stddev: 0.00000496629425340829",
            "extra": "mean: 72.75359832814009 usec\nrounds: 11365"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 847.5383195768218,
            "unit": "iter/sec",
            "range": "stddev: 0.00003282720046747523",
            "extra": "mean: 1.1798876545184442 msec\nrounds: 686"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 858.0745711812634,
            "unit": "iter/sec",
            "range": "stddev: 0.000016427717169657032",
            "extra": "mean: 1.1653998773362504 msec\nrounds: 856"
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
          "id": "03330a4268b7677faa34c93f405be3371cd97f21",
          "message": "perf(cypher): correlated-equality pushdown + optimizer for add_connections (0.8.4)\n\nPush `WHERE cur.prop = prior.other_prop` onto the current MATCH as a new\nEqualsNodeProp matcher resolved per-row from the bound node's property,\nso indexed lookups fire instead of cross-product + post-filter. Also\npush `cur.prop = scalar_var` (from WITH/UNWIND) as EqualsVar. WHERE\nstays as safety net.\n\nadd_connections(query=...) now runs cypher::optimize — previously the\nquery path skipped the entire planner (no pushdowns, no spatial fusion,\nno LIMIT/DISTINCT pushdown).\n\nSodir prospect derived connections: 29.6 s → 3.5 s (~8.5x).\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T12:54:33+02:00",
          "tree_id": "d5a6da8c734ed97882a2b2810d6fcb750bcccc89",
          "url": "https://github.com/kkollsga/kglite/commit/03330a4268b7677faa34c93f405be3371cd97f21"
        },
        "date": 1776596269945,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1139.7895883311878,
            "unit": "iter/sec",
            "range": "stddev: 0.000027292247804910373",
            "extra": "mean: 877.3549172914805 usec\nrounds: 399"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 789.6329272589771,
            "unit": "iter/sec",
            "range": "stddev: 0.00009358484018228559",
            "extra": "mean: 1.266411221567548 msec\nrounds: 677"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13690.074560600337,
            "unit": "iter/sec",
            "range": "stddev: 0.000006371588312689967",
            "extra": "mean: 73.04562115957884 usec\nrounds: 6021"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1650.530678344644,
            "unit": "iter/sec",
            "range": "stddev: 0.00003248503780807622",
            "extra": "mean: 605.8657455570129 usec\nrounds: 900"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 643547.0661262288,
            "unit": "iter/sec",
            "range": "stddev: 4.2954991086347436e-7",
            "extra": "mean: 1.5538879013455944 usec\nrounds: 58056"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 130523.92851366274,
            "unit": "iter/sec",
            "range": "stddev: 9.026217029074785e-7",
            "extra": "mean: 7.661430447179069 usec\nrounds: 11495"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2241.5137820396526,
            "unit": "iter/sec",
            "range": "stddev: 0.000010381727437590373",
            "extra": "mean: 446.1270807311547 usec\nrounds: 3332"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1617.3926570754602,
            "unit": "iter/sec",
            "range": "stddev: 0.000020821873075763073",
            "extra": "mean: 618.2790527862181 usec\nrounds: 1023"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13529.82175421007,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035408015014705885",
            "extra": "mean: 73.91080371689526 usec\nrounds: 9848"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 805.8225667896584,
            "unit": "iter/sec",
            "range": "stddev: 0.00001730134084123188",
            "extra": "mean: 1.240967976342399 msec\nrounds: 634"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 813.1974980095696,
            "unit": "iter/sec",
            "range": "stddev: 0.00002850497608649812",
            "extra": "mean: 1.2297135719768681 msec\nrounds: 778"
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
          "id": "34358abe27c9d10a1d2fd2333bab1bece9c666ed",
          "message": "chore: release 0.8.5\n\nInternal: test coverage, SAFETY docs, storage module reorganization.\nNo user-visible changes.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T17:27:38+02:00",
          "tree_id": "eb5b03931d5e541c841b31bd7e3d23bf5468ce43",
          "url": "https://github.com/kkollsga/kglite/commit/34358abe27c9d10a1d2fd2333bab1bece9c666ed"
        },
        "date": 1776613932888,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1181.394382067353,
            "unit": "iter/sec",
            "range": "stddev: 0.000023192501390065792",
            "extra": "mean: 846.4573855938554 usec\nrounds: 472"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 805.4376167427943,
            "unit": "iter/sec",
            "range": "stddev: 0.00003288748449708893",
            "extra": "mean: 1.2415610833325863 msec\nrounds: 696"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13938.060771988188,
            "unit": "iter/sec",
            "range": "stddev: 0.000003506833210800995",
            "extra": "mean: 71.74599224088155 usec\nrounds: 6444"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1672.0780007491048,
            "unit": "iter/sec",
            "range": "stddev: 0.000021460188951264863",
            "extra": "mean: 598.0582242885748 usec\nrounds: 914"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 650510.8540308799,
            "unit": "iter/sec",
            "range": "stddev: 3.8854857786383384e-7",
            "extra": "mean: 1.5372533660330434 usec\nrounds: 63280"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 132377.29939083534,
            "unit": "iter/sec",
            "range": "stddev: 8.763346367490666e-7",
            "extra": "mean: 7.554165288170484 usec\nrounds: 16565"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2132.2875289327735,
            "unit": "iter/sec",
            "range": "stddev: 0.000014506630348867668",
            "extra": "mean: 468.97990370956575 usec\nrounds: 3666"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1632.938795192878,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199071090251583",
            "extra": "mean: 612.3928238730363 usec\nrounds: 1198"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13642.681568681362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000370919635881682",
            "extra": "mean: 73.29937263181723 usec\nrounds: 10187"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 806.7503183873266,
            "unit": "iter/sec",
            "range": "stddev: 0.000028848854123933736",
            "extra": "mean: 1.2395408805031212 msec\nrounds: 636"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 811.1411397305615,
            "unit": "iter/sec",
            "range": "stddev: 0.000056972324051758527",
            "extra": "mean: 1.2328310709677137 msec\nrounds: 775"
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
          "id": "02af07a3e71b368659d024e8787ec813e6db0671",
          "message": "perf(describe): inverted-index fast path for connections=['T'] (0.8.6)\n\nRewrote write_connections_detail to stop sweeping edge_references\nthree times per topic. The old path materialised every visited edge\ninto the DiskGraph per-query arena, so describe(connections=['P31'])\non Wikidata (863M edges) exhausted VM and was SIGKILLed.\n\nNew path, in order: pair counts come from type_connectivity_cache\nwhen populated; property-stats scan is skipped when metadata declares\nno edge properties; only matching edges are walked via the persisted\nconn_type_index_* inverted index, capped at two samples via an\nearly-exit callback. Nothing materialises EdgeData on disk.\n\nMeasured on Wikidata (122M nodes, 863M edges, cold page cache):\n  describe(connections=['P170'])  1.3M edges: 108s  -> 0.24s\n  describe(connections=['P31'])  122M edges: SIGKILL -> 0.25s\n  describe(connections=True):     unchanged at 0.15s\n\nNew API:\n  GraphBackend::for_each_edge_of_conn_type(conn, |src,tgt,idx,props|)\n    returns bool; on disk uses the inverted index, on memory/mapped\n    filters petgraph's resident edge_references.\n  DiskGraph::edge_properties_at(edge_idx) borrows a property slice\n    without pushing into the arena.\n  describe(..., max_pairs=N) caps the (src_type, tgt_type) pair\n    breakdown (default 50). P31 has 191k distinct pairs — the cap\n    keeps the default response at ~4KB with a <more pairs=\"…\"\n    edges=\"…\"/> marker for the tail; pass max_pairs=1000 to drill in.",
          "timestamp": "2026-04-19T20:17:16+02:00",
          "tree_id": "03b64648792feae9958ed55e3320bb482c70ea05",
          "url": "https://github.com/kkollsga/kglite/commit/02af07a3e71b368659d024e8787ec813e6db0671"
        },
        "date": 1776622987135,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1046.849443787546,
            "unit": "iter/sec",
            "range": "stddev: 0.000019349364469876087",
            "extra": "mean: 955.2472000003718 usec\nrounds: 500"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 748.6317226996388,
            "unit": "iter/sec",
            "range": "stddev: 0.00005833691295716892",
            "extra": "mean: 1.3357702721892453 msec\nrounds: 676"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 12603.620521117355,
            "unit": "iter/sec",
            "range": "stddev: 0.000006253362703202482",
            "extra": "mean: 79.34228092035148 usec\nrounds: 6824"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1681.4375986707314,
            "unit": "iter/sec",
            "range": "stddev: 0.00004393298401576098",
            "extra": "mean: 594.7291774553839 usec\nrounds: 896"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 665375.0919230345,
            "unit": "iter/sec",
            "range": "stddev: 4.585244510246316e-7",
            "extra": "mean: 1.5029116841597556 usec\nrounds: 76249"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 126678.26832239497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013407325193434232",
            "extra": "mean: 7.894013813442805 usec\nrounds: 25772"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2161.379954187894,
            "unit": "iter/sec",
            "range": "stddev: 0.000014678884181833722",
            "extra": "mean: 462.66737972765884 usec\nrounds: 3305"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1676.7475872858572,
            "unit": "iter/sec",
            "range": "stddev: 0.00008458403654372023",
            "extra": "mean: 596.3926875950929 usec\nrounds: 1306"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 14121.536375534215,
            "unit": "iter/sec",
            "range": "stddev: 0.000005614626204398014",
            "extra": "mean: 70.81382460144464 usec\nrounds: 11357"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 859.3801172605931,
            "unit": "iter/sec",
            "range": "stddev: 0.00002855558006933432",
            "extra": "mean: 1.1636294346530316 msec\nrounds: 681"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 882.5936682064075,
            "unit": "iter/sec",
            "range": "stddev: 0.00001902864308087805",
            "extra": "mean: 1.133024217171401 msec\nrounds: 792"
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
          "id": "412cf0ae0f81f2649c3aa0ac2cc725918d84d7d9",
          "message": "feat(cypher): disk property index + schema validation + diagnostics (0.8.7)\n\nAgent-usability overhaul for Cypher on disk-backed graphs.\n\n- Persistent disk-backed property index: create_index on storage='disk'\n  writes four mmap'd files (property_index_{type}_{property}_*.bin),\n  lazy-loaded on first query after reopen. Equality + STARTS WITH\n  served by the same sorted-string layout. Planner pushdown via\n  GraphRead::lookup_by_property_{eq,prefix} trait methods; matcher\n  falls through to the disk index after the in-memory property_indices\n  HashMap misses. Replaces the silent-OOM-on-load path for large\n  disk-backed types.\n- Schema validation at plan time: catches unknown pattern-literal\n  property names ({agee: 30}) before the executor scans, with \"Did\n  you mean?\" suggestions. Node/connection types and WHERE/RETURN\n  expression property accesses pass through (legitimate existence\n  checks, virtual columns).\n- Backend-aware default timeouts: Disk 10s, Mapped 60s, Memory none.\n  timeout_ms=0 is the documented escape hatch. Bare \"Query timed out\"\n  replaced with a structured hint about anchoring and timeout override.\n- Deadline polling added to three unanchored matcher scan loops; worst-\n  case overshoot drops from 20-60s to ~ms past the deadline.\n- ResultView.diagnostics: always-on elapsed_ms / timed_out / timeout_ms\n  dict attached to every cypher() result.\n- describe() annotates indexed properties with indexed=\"eq,prefix\"\n  (strings) or indexed=\"eq\" (numeric), plus an <indexing> hint in\n  <extensions> teaching agents to anchor.\n- MCP cypher_query tool accepts timeout_ms for per-call overrides.\n\nIn-memory backend: zero behaviour change. Disk: create_index now\npersists, default deadline applies, and structured diagnostics ship\nwith every result.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T22:24:04+02:00",
          "tree_id": "6e50ad2fb2843e8c858770f23a1efe82479cfe14",
          "url": "https://github.com/kkollsga/kglite/commit/412cf0ae0f81f2649c3aa0ac2cc725918d84d7d9"
        },
        "date": 1776630446683,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_nodes",
            "value": 1185.61251412053,
            "unit": "iter/sec",
            "range": "stddev: 0.00002320216472185778",
            "extra": "mean: 843.4458881718074 usec\nrounds: 465"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_add_connections",
            "value": 816.628950913749,
            "unit": "iter/sec",
            "range": "stddev: 0.00003000038490363141",
            "extra": "mean: 1.2245463485969632 msec\nrounds: 677"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_match",
            "value": 13771.412616666214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033825670819924663",
            "extra": "mean: 72.61419201032407 usec\nrounds: 5432"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_cypher_where",
            "value": 1644.4652823971978,
            "unit": "iter/sec",
            "range": "stddev: 0.000020984011144684874",
            "extra": "mean: 608.1004024251962 usec\nrounds: 907"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_traversal",
            "value": 663918.5422715864,
            "unit": "iter/sec",
            "range": "stddev: 4.100472209466412e-7",
            "extra": "mean: 1.5062088740261967 usec\nrounds: 69731"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_shortest_path",
            "value": 128808.02359284091,
            "unit": "iter/sec",
            "range": "stddev: 9.036817215786217e-7",
            "extra": "mean: 7.763491528765135 usec\nrounds: 15051"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_enable",
            "value": 2261.2698820651526,
            "unit": "iter/sec",
            "range": "stddev: 0.000009847422543916464",
            "extra": "mean: 442.22938974746734 usec\nrounds: 3687"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_where",
            "value": 1626.3996612834676,
            "unit": "iter/sec",
            "range": "stddev: 0.000020477324246405164",
            "extra": "mean: 614.8550222955982 usec\nrounds: 1211"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_cypher_match",
            "value": 13553.963231265558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031780987311255793",
            "extra": "mean: 73.77915838617987 usec\nrounds: 9963"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_columnar_save_kgl",
            "value": 797.3490029841016,
            "unit": "iter/sec",
            "range": "stddev: 0.000026212275097326777",
            "extra": "mean: 1.2541559546164494 msec\nrounds: 639"
          },
          {
            "name": "tests/benchmarks/test_bench_core.py::test_bench_save_v3",
            "value": 803.4585213753974,
            "unit": "iter/sec",
            "range": "stddev: 0.00008189114902014082",
            "extra": "mean: 1.2446193218389843 msec\nrounds: 783"
          }
        ]
      }
    ]
  }
}