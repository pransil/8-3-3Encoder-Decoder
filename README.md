## Readme.md
A simple encoder-decoder in pytorch
    Input: binary, one hot, 8 digits
    Hidden: 3
    Output: 8, one hot

I setup to easily vary the width of input and hidden layers. I found that with 8-wide input, it can train to 100% with as few as 2 hidden units but trains faster with 3.