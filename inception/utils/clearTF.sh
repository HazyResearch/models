#!/bin/bash
pkill python
ssh node001 'pkill python'
ssh node002 'pkill python'
ssh node003 'pkill python'
