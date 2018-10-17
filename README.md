# Composable Action-Conditioned Predictors: Flexible Off-Policy Learning for Robot Navigation

[arXiv paper link](https://arxiv.org/abs/1810.07167)

Click below to view the paper video:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lOLT7zifEkg/0.jpg)](https://www.youtube.com/watch?v=lOLT7zifEkg)

## Installation

Clone the repository and go into the folder:
```bash
$ git clone https://github.com/gkahn13/CAPs.git
$ cd CAPs
```
From now on, we will assume you are in the CAPs folder.

Create the data folder:
```bash
$ mkdir data
```
All experiments will be saved into the data folder. If you have an external hard drive that you wish to use for saving the data, you can create a symlink:
```bash
$ ln -s <path/to/external/harddrive/folder/> data
```

## Docker

For portability to different machine configurations, we will use docker to run the code. Make sure you have [docker](https://www.docker.com/) (with GPU support) installed. However, you do not need to know anything about how docker works.

Change into the docker directory, and build the dockerfile:
```bash
$ cd src/sandbox/caps/docker
$ ./caps-docker.sh build
```
The build will take a while the first time, but only needs to be done once.

After building, start docker and ssh into the session:
```bash
$ ./caps-docker.sh start
$ ./caps-docker.sh ssh
```

You will now notice you are inside the docker container. The repository is located at ~/caps (while other installed code is located in ~/source). The ~/caps folder in the docker container is the exact same as on your local machine, and changes can be made to the caps files either from inside the docker container or from your local machine.

To exit the docker container, ctrl-d. You can stop the container by:
```bash
$ ./caps-docker.sh stop
```
If you stop the container, you will have to start it again before you can ssh in.

From now on, we will assume you are ssh'd into the docker container and are in the ~/caps directory.

## Run pre-trained policy

We provide a pre-trained CAPs policy for the task of collision avoidance, driving at 7m/s, staying in the right lane, and driving towards a desired goal in the [CARLA](https://github.com/carla-simulator/carla/) simulator.

Download the pre-trained CAPs policy [here](https://drive.google.com/file/d/1uBWlM3WNoTQHaFPYUn79r87l0POGw-be/view?usp=sharing) and unzip it:
```bash
$ mkdir -p data/caps/carla/coll_speed_learnedroad_heading/offline
$ cd data/caps/carla/coll_speed_learnedroad_heading/offline
$ mv <path/to/caps.zip> .
$ unzip caps.zip
```

You can then run the CAPs policy:
```bash
$ cd scripts
$ python run_gcg_eval.py caps/carla/coll_speed_learnedroad_heading/offline/caps -itr 35 -numrollouts 15
```

Note: sometimes CARLA doesn't close when python finishes, in which case you should run
```bash
pkill -9 -f CarlaUE4
```

If you want to run goal-conditioned Q-learning (GC-DQL) or goal-conditioned Q-learning with separate value functions (GC-DQL-sep), similarly download this [file](https://drive.google.com/file/d/14snubzP1CMtx1JLX-YBVCCsftXnOEcSI/view?usp=sharing) and this [file](https://drive.google.com/file/d/1x1mTOzK4HuO3Ocm4VZo4j2o6bHEoM4fD/view?usp=sharing) and proceed similarly:
```bash
$ cd data/caps/carla/coll_speed_learnedroad_heading/offline
$ mv <path/to/gc_dql.zip> .
$ unzip gc_dql.zip
$ mv <path/to/gc_dql_sep.zip> .
$ unzip gc_dql_sep.zip

$ cd scripts
$ python run_gcg_eval.py caps/carla/coll_speed_learnedroad_heading/offline/gc_dql -itr 500 -numrollouts 15
$ python run_gcg_eval.py caps/carla/coll_speed_learnedroad_heading/offline/gc_dql_sep -itr 500 -numrollouts 15
```


## Run train policy

You can also train your own CAPs policy using the same data we gathered. Download this [file](https://drive.google.com/file/d/1Yg18qZqWpqWlvOUIGIyMhposbRQ7TDTH/view?usp=sharing) and extract it:
```bash
$ cd data/caps/carla/coll_speed_learnedroad_heading
$ mv <path/to/label_event_cues.zip>
$ unzip label_event_cues.zip
```

Next, delete the pre-trained policy folder from the previous section. Then train the CAPs policy:
```bash
$ cd scripts
$ python run_gcg_train.py caps/carla/coll_speed_learnedroad_heading/offline/caps
```

You can similarly train GC-DQL and GC-DQL-sep:
```bash
$ cd scripts
$ python run_gcg_train.py caps/carla/coll_speed_learnedroad_heading/offline/gc_dql
$ python run_gcg_train.py caps/carla/coll_speed_learnedroad_heading/offline/gc_dql_sep
```

## Run label data

You can also train and run the autonomous CAPs labeling system, which in this experiment is for the right lane event cue.

First, download the CARLA rollouts [here](https://drive.google.com/file/d/15x3RMAUV_Smx4nW6nj_69I70OKzCNNnJ/view?usp=sharing) and extract it:
```bash
$ mkdir -p data/caps/carla/coll_speed
$ cd data/caps/carla/coll_speed
$ mv <path/to/dql.zip>
$ unzip dql.zip
```
Warning: this file is very large (~300GB).

Then, train the right lane road segmenter:
```bash
$ cd src/sandbox/caps/scripts/carla
$ python learned_road_labeller.py train
```

After training (or if you stop it early), you can visualize the performance of the model:
```bash
$ python learned_road_labeller.py eval
```

Finally, you can then use this learned right lane labeller to label the training data (and convert it into the .tfrecord training data format):
```bash
$ cd scripts
$ python run_pkls_to_tfrecords.py caps/carla/coll_speed_learnedroad_heading/label_event_cues -pkl_folders caps/carla/coll_speed/dql -output_folder caps/carla/coll_speed_learnedroad_heading/label_event_cues
```

## Run gather data

You can also gather the CARLA data yourself. In our experiments, we used DQL to gather data for the task of collision avoidance:
```bash
$ cd scripts
$ python run_gcg.py caps/carla/coll_speed/dql
```

Warning: this will take multiple days to run.

## Referencing

If you use CAPs, please cite our CoRL 2018 paper.

```bib
@inproceedings{Kahn2018_CoRL,
  title = {Composable Action-Conditioned Predictors: Flexible Off-Policy Learning for Robot Navigation},
  author = {Gregory Kahn and Adam Villaflor and Pieter Abbeel and Sergey Levine},
  booktitle = {Proceedings of the 2nd Annual Conference on Robot Learning},
  year = {2018}
}
```
