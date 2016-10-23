# Deep Learning from Scratch (v2.0)

This course is organized by the DataScienceGroup@UB (http://cienciadedades.barcelona)

Deep learning is one of the fastest growing areas of machine learning and a hot topic in both academia and industry.
This course will cover the basics of deep learning by using a hands-on approach.

## Course Agenda

<li> Introduction to Deep Learning and its applications. Using the Jupyter notebook & Docker.
<li> Basic Concepts: Score & Loss functions, Optimization (SGD), Linear Regression.
<li> Automated differentiation, Backpropagation, Training a Neural Netwotk from Scratch.
<li> Tensorflow programming model. 
<li> Convolutions & CNN models.
<li> Recurrent Neural Netwoks.
<li> Unsupervised Learning.
<li> Advanced Applications: Neural art, colorization, music generation.

## Course Software Installation

The best way to run the course software is to use a Docker container. There’s full documentation on installing Docker at ``docker.com``, but in a few words, the steps are:

+ Go to ``docs.docker.com`` in your browser.
+ Step one of the instructions sends you to download the Docker Toolbox.
+ On the Toolbox page, click on the Mac/Windows download button.
+ Run that downloaded file to install the Toolbox.
+ At the end of the install process, chose the Docker Quickstart Terminal.
+ This opens up a new terminal window that runs through an installation script.
+ At the end of the script, you will see ASCII art of a whale and your are left at a prompt.
+ Run this command in the terminal: ``docker run hello-world``
+ This will give you output confirming your installation of docker has worked: ``Hello from Docker``
+ In the docker terminal, run (This operation requires a good internet connection to download ~2.5Gb; it will take some minutes):  ``docker pull deepub/deepub``    
+ Run the DeepUB image on your system: ``docker run -i -t -p 8888:8888 deepub/deepub``
+ Once these steps have been done, you can check the installation by starting your web browser and introducing this  URL: ``http://localhost:8888`` (or ``http://<DOCKER-MACHINE-IP>:8888`` if you are using a Docker Machine VM) to access to the fully operational Jupyter notebook of this course. (Note: the address of your Docker Machine VM is printed at the end of the Docker booting process, just after the ASCII whale).

## Docker Optimization 

(Source: ``https://petewarden.com/2016/02/28/tensorflow-for-poets/``)

Under the hood, Docker actually uses VirtualBox to run its images, and we’ll use its control panel to manage the setup. To do that, we’ll need to take the following steps:

+ Find the VirtualBox application on your Mac/PC. 
+ Once VirtualBox is open, you should see a left-hand pane showing virtual machines. There should be one called ``default`` that’s running.
+ Right-click on ``default`` to bring up the context menu and chose ``Close->ACPI Shutdown``. The other close options should also work, but this is the most clean.
+ Once the shutdown is complete, ``default`` should have the text ``Powered off`` below it. Right click on it again and choose ``Settings…`` from the menu.
+ Click on the ``System`` icon, and then choose the ``Motherboard`` tab.
+ Drag the ``Base Memory`` slider as far as the green section goes, which is normally around 75% of your total laptop’s memory. 
+ Click on the ``Processor`` tab, and set the number of processors higher than the default of 1. 
+ Click ``OK`` on the settings dialog.
+ Right-click on ``default`` and choose ``Start->Headless Start``.
