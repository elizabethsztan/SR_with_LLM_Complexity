#!/bin/bash

rsync -avz --progress \
  -e "ssh -J eszt2@ssh.maths.cam.ac.uk" \
  eszt2@swirles.maths.cam.ac.uk:/cephfs/home/eszt2/LLM_Complexity/ \
  /Users/liz/PhD/LLM_Complexity/SR_with_LLM_Complexity/

echo "Successfully synced Swirles -> Laptop"