# Cryptocurrency Trading Dashboard

Welcome to the CryptoDash project. This project was originally created for Differential Captial as an intern project. The original project was a dashboard ran through a Jupyter Notebook and was static. The project was stopped after the internship ended in Spring 2022. I decided to continue this project on my own and refactor the code for it to be usable as an open source dashboard. 

What I've added on my own time
1. A Dockerfile
2. Converted the notebook into a python script
3. Refactoring multiple functions and portions of the code for improved speed 
4. Deployment to AWS 

What I'm working on 
1. Adding functionality to the sentiment analysis via NLP research
2. Using Lambda to continously update the data

For personal use
1. Make sure to clone the repo. You really only need the dashboard folder
2. Make sure you have Docker installed
3. Navigate into the dashboard folder and run ``docker build -t <giveitaname> .``
4. Run ``docker run -p 127.0.0.1:8050:8050/tcp dashboard`` and then navigate to port 8050 to view  

The original dashboard is present in this repo for reference. If you'd like to view the original repository, please send me a message. As always, feel free to create a PR if you'd like to contribute and let me know if you'd like to collaborate
