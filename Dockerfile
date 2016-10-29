FROM gcr.io/tensorflow/tensorflow
   #Install packages
   RUN DEBIAN_FRONTEND=noninteractive apt-get update
   RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy install wget python-pip git
   RUN DEBIAN_FRONTEND=noninteractive pip install --upgrade pip
   RUN DEBIAN_FRONTEND=noninteractive pip install tqdm pandas seaborn bokeh sklearn keras 
   RUN DEBIAN_FRONTEND=noninteractive pip install git+https://github.com/tflearn/tflearn.git

   #Remove examples
   RUN rm -Rf *

   RUN git clone https://github.com/jvitria/DeepLearningBBVA2016
