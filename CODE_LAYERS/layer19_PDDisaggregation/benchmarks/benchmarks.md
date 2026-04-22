python -m sglang.launch_server     
--model Qwen/Qwen3-1.7B \
    --kv-cache-dtype auto
    --mem-fraction-static 0.75
    --port 30000     
    --host 127.0.0.1
     --context-len 1024

max_total_num_tokens=75115, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=4096, context_len=1024, available_gpu_mem=2.67 GB 
python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 4096     --random-output-len 4096     --num-prompts 10     --max-concurrency 1     --warmup-requests 2     --flush-cache     --output-file results_bf16.json

============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 1         
Successful requests:                     10        
Benchmark duration (s):                  348.95    
Total input tokens:                      21461     
Total input text tokens:                 21461     
Total generated tokens:                  22652     
Total generated tokens (retokenized):    22652     
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          61.50     
Output token throughput (tok/s):         64.91     
Peak output token throughput (tok/s):    70.00     
Peak concurrent requests:                2         
Total token throughput (tok/s):          126.42    
Concurrency:                             1.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   34892.83  
Median E2E Latency (ms):                 41072.73  
P90 E2E Latency (ms):                    55983.12  
P99 E2E Latency (ms):                    63000.92  
---------------Time to First Token----------------
Mean TTFT (ms):                          216.01    
Median TTFT (ms):                        195.57    
P99 TTFT (ms):                           391.30    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.08     
Median TPOT (ms):                        15.08     
P99 TPOT (ms):                           15.67     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           15.32     
Median ITL (ms):                         15.36     
P95 ITL (ms):                            15.99     
P99 ITL (ms):                            16.19     
Max ITL (ms):                            18.11     
==================================================

/home/arun/PROJECTS/sglang_development/.conda/bin/python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 16     --warmup-requests 5     --flush-cache     --output-file results_bf16.json
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 16        
Successful requests:                     300       
Benchmark duration (s):                  102.52    
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77287     
Request throughput (req/s):              2.93      
Input token throughput (tok/s):          749.45    
Output token throughput (tok/s):         753.86    
Peak output token throughput (tok/s):    862.00    
Peak concurrent requests:                25        
Total token throughput (tok/s):          1503.31   
Concurrency:                             15.41     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5265.90   
Median E2E Latency (ms):                 4959.65   
P90 E2E Latency (ms):                    9197.75   
P99 E2E Latency (ms):                    10393.90  
---------------Time to First Token----------------
Mean TTFT (ms):                          76.56     
Median TTFT (ms):                        67.50     
P99 TTFT (ms):                           338.32    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.24     
Median TPOT (ms):                        20.25     
P99 TPOT (ms):                           22.07     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           20.22     
Median ITL (ms):                         18.42     
P95 ITL (ms):                            37.55     
P99 ITL (ms):                            65.76     
Max ITL (ms):                            359.15    
==================================================




/home/arun/PROJECTS/sglang_development/.conda/bin/python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 24     --warmup-requests 5     --flush-cache     --output-file results_bf16.json



============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 24        
Successful requests:                     300       
Benchmark duration (s):                  78.59     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77287     
Request throughput (req/s):              3.82      
Input token throughput (tok/s):          977.71    
Output token throughput (tok/s):         983.46    
Peak output token throughput (tok/s):    1176.00   
Peak concurrent requests:                32        
Total token throughput (tok/s):          1961.17   
Concurrency:                             22.72     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5951.15   
Median E2E Latency (ms):                 5623.65   
P90 E2E Latency (ms):                    10441.88  
P99 E2E Latency (ms):                    11720.95  
---------------Time to First Token----------------
Mean TTFT (ms):                          93.06     
Median TTFT (ms):                        70.71     
P99 TTFT (ms):                           495.44    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.95     
Median TPOT (ms):                        23.07     
P99 TPOT (ms):                           25.66     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           22.83     
Median ITL (ms):                         20.23     
P95 ITL (ms):                            49.41     
P99 ITL (ms):                            69.43     
Max ITL (ms):                            578.57    
==================================================

/home/arun/PROJECTS/sglang_development/.conda/bin/python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 32     --warmup-requests 5     --flush-cache     --output-file results_bf16.json


============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 32        
Successful requests:                     300       
Benchmark duration (s):                  66.86     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77287     
Request throughput (req/s):              4.49      
Input token throughput (tok/s):          1149.25   
Output token throughput (tok/s):         1156.01   
Peak output token throughput (tok/s):    1470.00   
Peak concurrent requests:                41        
Total token throughput (tok/s):          2305.27   
Concurrency:                             29.72     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6623.29   
Median E2E Latency (ms):                 6293.59   
P90 E2E Latency (ms):                    11674.35  
P99 E2E Latency (ms):                    13302.04  
---------------Time to First Token----------------
Mean TTFT (ms):                          115.26    
Median TTFT (ms):                        75.26     
P99 TTFT (ms):                           788.86    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.56     
Median TPOT (ms):                        25.74     
P99 TPOT (ms):                           29.18     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           25.36     
Median ITL (ms):                         22.06     
P95 ITL (ms):                            59.69     
P99 ITL (ms):                            72.77     
Max ITL (ms):                            793.99    
==================================================


/home/arun/PROJECTS/sglang_development/.conda/bin/python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 48     --warmup-requests 5     --flush-cache     -
-output-file results_bf16.json



============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 48        
Successful requests:                     300       
Benchmark duration (s):                  53.86     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77287     
Request throughput (req/s):              5.57      
Input token throughput (tok/s):          1426.54   
Output token throughput (tok/s):         1434.93   
Peak output token throughput (tok/s):    1822.00   
Peak concurrent requests:                57        
Total token throughput (tok/s):          2861.47   
Concurrency:                             43.34     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   7780.74   
Median E2E Latency (ms):                 7513.06   
P90 E2E Latency (ms):                    13606.19  
P99 E2E Latency (ms):                    15639.04  
---------------Time to First Token----------------
Mean TTFT (ms):                          169.73    
Median TTFT (ms):                        79.99     
P99 TTFT (ms):                           1080.44   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.09     
Median TPOT (ms):                        30.63     
P99 TPOT (ms):                           37.42     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           29.66     
Median ITL (ms):                         25.32     
P95 ITL (ms):                            65.98     
P99 ITL (ms):                            89.53     
Max ITL (ms):                            1089.36   
==================================================


python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 64     --warmup-requests 5     --flush-cache     -
-output-file results_bf16.json



============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     300       
Benchmark duration (s):                  47.40     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77288     
Request throughput (req/s):              6.33      
Input token throughput (tok/s):          1621.10   
Output token throughput (tok/s):         1630.63   
Peak output token throughput (tok/s):    2208.00   
Peak concurrent requests:                75        
Total token throughput (tok/s):          3251.73   
Concurrency:                             56.14     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8870.07   
Median E2E Latency (ms):                 8537.08   
P90 E2E Latency (ms):                    15394.34  
P99 E2E Latency (ms):                    18193.56  
---------------Time to First Token----------------
Mean TTFT (ms):                          240.76    
Median TTFT (ms):                        86.54     
P99 TTFT (ms):                           1449.84   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.27     
Median TPOT (ms):                        35.32     
P99 TPOT (ms):                           47.04     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           33.63     
Median ITL (ms):                         28.15     
P95 ITL (ms):                            75.14     
P99 ITL (ms):                            107.66    
Max ITL (ms):                            1387.40   
==================================================

python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 600     --max-concurrency 128     --warmup-requests 5     --flush-cache     --output-file results_bf16.json


============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 128       
Successful requests:                     600       
Benchmark duration (s):                  73.08     
Total input tokens:                      154124    
Total input text tokens:                 154124    
Total generated tokens:                  156584    
Total generated tokens (retokenized):    156575    
Request throughput (req/s):              8.21      
Input token throughput (tok/s):          2108.93   
Output token throughput (tok/s):         2142.59   
Peak output token throughput (tok/s):    3289.00   
Peak concurrent requests:                142       
Total token throughput (tok/s):          4251.52   
Concurrency:                             115.79    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   14103.83  
Median E2E Latency (ms):                 14145.28  
P90 E2E Latency (ms):                    24379.00  
P99 E2E Latency (ms):                    28958.37  
---------------Time to First Token----------------
Mean TTFT (ms):                          402.31    
Median TTFT (ms):                        110.41    
P99 TTFT (ms):                           2603.82   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          54.82     
Median TPOT (ms):                        55.37     
P99 TPOT (ms):                           100.37    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           52.92     
Median ITL (ms):                         43.01     
P95 ITL (ms):                            107.00    
P99 ITL (ms):                            166.50    
Max ITL (ms):                            2533.16   
==================================================








/home/arun/PROJECTS/sglang_development/.conda/bin/python -m sglang.launch_server \
    --model Qwen/Qwen3-1.7B \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.75 \
    --port 30000 \
    --host 127.0.0.1


[2026-04-21 20:58:20] max_total_num_tokens=149848, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=2048, context_len=40960, available_gpu_mem=2.33 GB



python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 64     --warmup-requests 5     --flush-cache     --output-file results_bf16.json


============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     300       
Benchmark duration (s):                  41.59     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77284     
Request throughput (req/s):              7.21      
Input token throughput (tok/s):          1847.40   
Output token throughput (tok/s):         1858.26   
Peak output token throughput (tok/s):    2683.00   
Peak concurrent requests:                78        
Total token throughput (tok/s):          3705.66   
Concurrency:                             55.79     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   7734.37   
Median E2E Latency (ms):                 7447.02   
P90 E2E Latency (ms):                    13386.77  
P99 E2E Latency (ms):                    15812.80  
---------------Time to First Token----------------
Mean TTFT (ms):                          238.46    
Median TTFT (ms):                        79.57     
P99 TTFT (ms):                           1396.99   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.79     
Median TPOT (ms):                        30.67     
P99 TPOT (ms):                           41.02     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           29.21     
Median ITL (ms):                         23.34     
P95 ITL (ms):                            71.79     
P99 ITL (ms):                            107.67    
Max ITL (ms):                            1392.27   
==================================================
python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 96     --warmup-requests 5     --flush-cache     --output-file results_bf16.json


============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 96        
Successful requests:                     300       
Benchmark duration (s):                  34.99     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77287     
Request throughput (req/s):              8.57      
Input token throughput (tok/s):          2195.92   
Output token throughput (tok/s):         2208.84   
Peak output token throughput (tok/s):    3371.00   
Peak concurrent requests:                109       
Total token throughput (tok/s):          4404.76   
Concurrency:                             79.83     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   9310.54   
Median E2E Latency (ms):                 9177.03   
P90 E2E Latency (ms):                    16234.37  
P99 E2E Latency (ms):                    19776.39  
---------------Time to First Token----------------
Mean TTFT (ms):                          447.27    
Median TTFT (ms):                        94.19     
P99 TTFT (ms):                           2067.48   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          35.76     
Median TPOT (ms):                        36.74     
P99 TPOT (ms):                           58.48     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           34.71     
Median ITL (ms):                         26.93     
P95 ITL (ms):                            77.92     
P99 ITL (ms):                            125.00    
Max ITL (ms):                            1930.43   
==================================================


python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 300     --max-concurrency 128     --warmup-requests 5     --flush-cache     --output-file results_bf16.json



============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 128       
Successful requests:                     300       
Benchmark duration (s):                  31.54     
Total input tokens:                      76836     
Total input text tokens:                 76836     
Total generated tokens:                  77288     
Total generated tokens (retokenized):    77286     
Request throughput (req/s):              9.51      
Input token throughput (tok/s):          2435.80   
Output token throughput (tok/s):         2450.12   
Peak output token throughput (tok/s):    3964.00   
Peak concurrent requests:                140       
Total token throughput (tok/s):          4885.92   
Concurrency:                             101.26    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10647.02  
Median E2E Latency (ms):                 10405.16  
P90 E2E Latency (ms):                    18830.64  
P99 E2E Latency (ms):                    22650.24  
---------------Time to First Token----------------
Mean TTFT (ms):                          699.34    
Median TTFT (ms):                        115.43    
P99 TTFT (ms):                           2687.05   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          40.62     
Median TPOT (ms):                        41.52     
P99 TPOT (ms):                           63.98     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           38.95     
Median ITL (ms):                         29.76     
P95 ITL (ms):                            89.51     
P99 ITL (ms):                            135.76    
Max ITL (ms):                            2595.86   
==================================================

python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 512     --random-output-len 512     --num-prompts 600     --max-concur
rency 128     --warmup-requests 5     --flush-cache     --output-file results_bf16.json



============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 128       
Successful requests:                     600       
Benchmark duration (s):                  60.50     
Total input tokens:                      154124    
Total input text tokens:                 154124    
Total generated tokens:                  156584    
Total generated tokens (retokenized):    156576    
Request throughput (req/s):              9.92      
Input token throughput (tok/s):          2547.52   
Output token throughput (tok/s):         2588.18   
Peak output token throughput (tok/s):    3967.00   
Peak concurrent requests:                145       
Total token throughput (tok/s):          5135.70   
Concurrency:                             115.26    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   11622.39  
Median E2E Latency (ms):                 11649.31  
P90 E2E Latency (ms):                    20017.68  
P99 E2E Latency (ms):                    23857.87  
---------------Time to First Token----------------
Mean TTFT (ms):                          393.47    
Median TTFT (ms):                        98.01     
P99 TTFT (ms):                           2623.89   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.17     
Median TPOT (ms):                        45.31     
P99 TPOT (ms):                           88.35     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           43.34     
Median ITL (ms):                         32.17     
P95 ITL (ms):                            98.16     
P99 ITL (ms):                            153.88    
Max ITL (ms):                            2557.60   
==================================================

python -m sglang.bench_serving     --backend sglang-oai     --host 127.0.0.1     --port 30000     --model Qwen/Qwen3-1.7B     --tokenizer Qwen/Qwen3-1.7B     --dataset-name random     --random-input-len 4096     --random-output-len 4096     --num-prompts 10     --m
ax-concurrency 1     --warmup-requests 2     --flush-cache     --output-file results
_bf16.json


============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 1         
Successful requests:                     10        
Benchmark duration (s):                  365.34    
Total input tokens:                      21461     
Total input text tokens:                 21461     
Total generated tokens:                  22652     
Total generated tokens (retokenized):    22652     
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          58.74     
Output token throughput (tok/s):         62.00     
Peak output token throughput (tok/s):    70.00     
Peak concurrent requests:                2         
Total token throughput (tok/s):          120.74    
Concurrency:                             1.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   36531.42  
Median E2E Latency (ms):                 42598.85  
P90 E2E Latency (ms):                    57711.40  
P99 E2E Latency (ms):                    67615.27  
---------------Time to First Token----------------
Mean TTFT (ms):                          214.21    
Median TTFT (ms):                        192.75    
P99 TTFT (ms):                           398.22    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.61     
Median TPOT (ms):                        15.50     
P99 TPOT (ms):                           16.87     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           16.04     
Median ITL (ms):                         16.06     
P95 ITL (ms):                            17.35     
P99 ITL (ms):                            17.75     
Max ITL (ms):                            20.11     
==================================================