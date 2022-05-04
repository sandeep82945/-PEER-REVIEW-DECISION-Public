# -PEER-REVIEW-DECISION-Public

# PEER-REVIEW-DECISION

Here we propose a model that could help you determine the acceptance/rejection of a paper using peer reviews.

We use the ASAP dataset as part of our training and evaluation
It can be downloaded directly from the link below:

https://drive.google.com/file/d/1nJdljy468roUcKLbVwWUhMs7teirah75/view?usp=sharing

The files from  the above link will be downloaded and stored in a folder named dataset.

After this follow the steps below to run the model

Step 1 -Install all the dependicies required for running our model using the following command.

    pip install -r requirements.txt

Step -2 This will create the files in accordance with the needs of the model. The model will be stored  in a file named input_files.

    python create_data.py
   
Step -3 Now run the model code by running the main.py file

    python main.py 
