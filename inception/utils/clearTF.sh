#!/bin/bash
pkill -u daniter python
ssh raiders3 'pkill -u daniter python'
ssh raiders8 'pkill -u daniter python'
ssh raiders2 'pkill -u daniter python'
