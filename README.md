# UCI-MDS-F25-Soccer

UC Irvine MDS Fall 2025 Soccer Capstone

If you're here, then welcome to our repository! There's still a lot of work to be done, but maybe you're here to continue this project.

Contents
- [Repository Explanation](#repository-explanation)
	- [The Notebooks Folder](#the-notebooks-folder)
- [Setup with uv](#setup-with-uv)
- [About the team](#about-the-team)

## Repository Explanation
To begin with, here is a quick rundown of our repository:  
`Streamlit/` : Includes Streamlit app script and relevant scripts. Currently does not implement anything from `scripts/` (this could be future work).  
`imgs/` : Includes flag images and an image of a field. The field image might be replaceable with a library such as `mplsoccer`.  
`metrics/` : Includes notebooks and scripts related to xG, xT, PPDA, and Field Tilt metric calculations. Many of the final outputs are saved as csv files; future work could include converting these into scripts (unsure if this is done by us as of writing this README) and using the scripts to dynamically update the streamlit app.  
`notebooks/` : I'll come back to this, but it has exploration notebooks.  
`scripts/` : Has script (`*.py`) files for computing metrics.

One folder that is referenced in various files but not included with any of the git pushes is a `statsbomb/` folder which includes `DATABASE_SPECIFICATION.md` and `statsbomb_euro2020.db`. This is different from the layout described in `metrics/data_processing.py`, but if you're cloning the repo to continue this project, try having consistency across the group lol.

### The Notebooks Folder
There is a bit in each notebook.  
- To begin with, `exploration.ipynb` has a lot of random stuff to begin with (currently it has my tests from writing `scripts/compute_xT.py`) but ends with a visualization of how long the teams lasted in the Euro UEFA 2020 tournament.  
- `change_pt_detection.ipynb` and `change_pt_example.ipynb` both have examples of **change point detection**. This is something that we wanted to implement but ended up abandoning due to 1) limited data, 2) limited time, and 3) too much hyperparameter tuning. The original goal was to find change points in a game or change points across a season based on xG (very few shots per game, so it doesn't work) or other metrics (e.g., passes per 5 minute interval). You might also need to deal with the assumption that the data is Gaussian at some point. Finally, the event correlation in `change_pt_example.ipynb` is an initial pass at trying to give "importance" to events that occur near a detected change point. This could be future work.  
- `Heatmap.ipynb` has the initial pass of creating heatmaps for the streamlit app.  
- `kloppy_exploration.ipynb` includes an exploration of the `kloppy` API as a data source, where it largely follows the getting-started guide for Kloppy. Enjoy my crashout at the broken documentation; I have working code for you. Future work could include integrating the `kloppy` API to dynamically call more data as needed for the dashboard or for training the models (xG, xT). You could also replace our heatmap code with "prettier" (maybe) heatmaps from `mplsoccer`.

And that's it! Now you should be equipped to explore our repository.

## Setup with uv
If you're comfortable using virtual environments, feel free to switch to your preferred package manager. You could also just `pip install` every package in `pyproject.toml`. Otherwise, consider using uv instead of conda! It's extremely fast (written in Rust), and I (Tim) have been using it since July 2025 and will never return to the dark days of conda.

To begin with, download uv at [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/). In a terminal (git bash or MSYS2 on Windows), navigate to this directory and enter the command
```
uv sync
```
to create the virtual environment and install the packages locally. Then, when you're ready to activate the virtual environment, run
```
source .venv/Scripts/activate
```
If you're running on a Mac or Linux machine, it might be
```
source .venv/bin/activate
```

And run python scripts from there. I'm not sure how to get uv working in a notebook, but I'm sure you can figure it out.

## About the team

We are all Master of Data Science students at the University of California, Irvine (graduating in December 2025), but here's a bit about the individual members.

<img src="https://raw.githubusercontent.com/timng-gnmit/timng-gnmit/refs/heads/main/tim.png" alt="Timothy Ng" width="200"/>  

**Name:** Timothy Ng  
**Undergraduate Experience:** Math @ University of California, Davis  
**About Me:** I love making coffee and cooking! I try to make latte art every morning, and I've been writing a recipe book that currently has over 35 recipes.  
**LinkedIn:** <https://www.linkedin.com/in/timothy-ng-data/>

![Ziwen Zhai](https://github.com/user-attachments/assets/0c091c52-1acc-4e71-adfb-0e335aa37156)

**Name:** Ziwen Zhai  
**Undergraduate Experience:** Data Analytics @ Washington State University  
**About Me:** I enjoy playing basketball and soccer — sports keep me active and energized. I'm also a big fan of rock music, especially Linkin Park.  
**LinkedIn:** <https://www.linkedin.com/in/ziwen-zhai/>

**Name:** Indrajeet Patwardhan  
**Undergraduate Experience:** Computer Science @ California State University, Fullerton  
**About Me:** I enjoy going to the gym and playing pickleball. I have been practicing martial arts for the last 10 years, and I like listening to EDM.  
**LinkedIn:** <https://www.linkedin.com/in/indrajeet-patwardhan-163b08211/>

**Name:** Bryan Torres  
**Undergraduate Experience:** Applied Math @ Cal Poly, Pomona  
**About Me:** I like to exercise and go on walks with my dogs.  
**LinkedIn:** <https://www.linkedin.com/in/bryantorres-okok/>

**Name:** Yue Xu
**Undergraduate Experience:** Information System @ NUAA  
**About Me:** I like to watch movies and play games.  
**LinkedIn:** <https://www.linkedin.com/in/yuexu4541/>

**Name:** Luyao Wang  
**Undergraduate Experience:** Data Science @ Duke Kunshan University  
**About Me:** I like to watch anime and play games.  
**LinkedIn:** <https://www.linkedin.com/in/1uyao-wang/>
