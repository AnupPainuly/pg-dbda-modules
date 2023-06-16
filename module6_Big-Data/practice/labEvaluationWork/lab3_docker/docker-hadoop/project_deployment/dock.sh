#!/bin/bash
sudo docker cp ./hire_data/ namenode:/tmp/
sudo docker cp ./lab3/ namenode:/tmp/
sudo docker cp ./lab3_workflow.sh namenode:/tmp/
sudo docker exec -it namenode bash

