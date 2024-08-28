#!/bin/bash

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k64m64c.py"
python3.11 test10k64m64c.py > test10k64m64c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k64m128c.py"
python3.11 test10k64m128c.py > test10k64m128c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k64m256c.py"
python3.11 test10k64m256c.py > test10k64m256c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k64m512c.py"
python3.11 test10k64m512c.py > test10k64m512c.txt
echo run time is $(expr `date +%s` - $start) s


start=`date +%s.%N`
date
tput setaf 32; echo "running test10k128m64c.py"
python3.11 test10k128m64c.py > test10k128m64c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k128m128c.py"
python3.11 test10k128m128c.py > test10k128m128c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k128m256c.py"
python3.11 test10k128m256c.py > test10k128m256c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k128m512c.py"
python3.11 test10k128m512c.py > test10k128m512c.txt
echo run time is $(expr `date +%s` - $start) s


start=`date +%s.%N`
date
tput setaf 32; echo "running test10k256m64c.py"
python3.11 test10k256m64c.py > test10k256m64c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k256m128c.py"
python3.11 test10k256m128c.py > test10k256m128c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k256m256c.py"
python3.11 test10k256m256c.py > test10k256m256c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k256m512c.py"
python3.11 test10k256m512c.py > test10k256m512c.txt
echo run time is $(expr `date +%s` - $start) s


start=`date +%s.%N`
date
tput setaf 32; echo "running test10k512m64c.py"
python3.11 test10k512m64c.py > test10k512m64c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k512m128c.py"
python3.11 test10k512m128c.py > test10k512m128c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k512m256c.py"
python3.11 test10k512m256c.py > test10k512m256c.txt
echo run time is $(expr `date +%s` - $start) s

start=`date +%s.%N`
date
tput setaf 32; echo "running test10k512m512c.py"
python3.11 test10k512m512c.py > test10k512m512c.txt
echo run time is $(expr `date +%s` - $start) s
