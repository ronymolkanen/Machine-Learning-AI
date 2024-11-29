# Machine-Learning-AI

* Solutions in Pattern Recognition folder
  
  * TreeRecognitionProject.ipynb -file
    * As a group we created a dataset which consists overall 136 pictures of different tree species (Maple, Juniper, Birch, Pine, Spruce)
      * Dataset was split 106 pictures for training / 30 pictures for validation
        
    * For Machine Learning part we used pre-trained ResNet18 model and added some more training into model of our specific purposes
      * Our model predicts 25/30 of validation data right   
    * Finally we created Gradio interface for live classification tasks
   
* AI folder
 
  * 2048 folder
   * Scope of this project was to develop as efficient AI as possible to maximise the score in 2048 game.
     * AI_heuristics.py -file contains different heuristics to make the game more efficient
     *  constants.py -file contains the game board ui
     *  logic.py -file contains the basic game logic
     *  puzzle.py -file is the main program
    
  * A_Star_Final.ipynb -file
    * In this project we created an ai which finds the shortest path in labyrinth. Movement is only allowed forward and right.
    * We utilized the A* algorithm and added some heuristics

* AI_App folder
  
  * We created AI to summarize text in english and also translate it into finnish if wanted.
  * We used pre-trained language model BART for English summarizing and Helsinki-NLP for translating into Finnish
  * And finally created Gradio interface for live summarizing
      


