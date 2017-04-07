<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`tag.lua` options:


-h
:    This help.

-md
:    Dump help in Markdown format.

-config <string\>
:    Load options from this file.

-save_config <string\>
:    Save options to this file.

## Data options


-src <string\>
:    Source sequences to tag.

-output <string\>
:    (default: `pred.txt`)<br/>Output file.

## Tagger options


-model <string\>
:    Path to the serialized model file.

-batch_size <number\>
:    (default: `30`)<br/>Batch size.

## Cuda options


-gpuid <string\>
:    (default: `0`)<br/>List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.

-fallback_to_cpu
:    If GPU can't be used, rollback on the CPU.

-fp16
:    Use half-precision float on GPU.

-no_nccl
:    Disable usage of nccl in parallel mode.

## Logger options


-log_file <string\>
:    Output logs to a file under this path instead of stdout.

-disable_logs
:    If set, output nothing.

-log_level <string\>
:    (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options


-time
:    Measure average translation time.

