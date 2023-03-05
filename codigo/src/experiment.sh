#!/bin/bash
command rm *log
luxai-s2 -o cli_replay.html -v 3 -s 41 main.py main.py > stdout.log
