# -*- coding: utf-8 -*-
"""Using Python for Research HW3-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16FKOObNHvc6uayDG5Kcgg7sOkpQ3Sqvp

# Using Python for Research Homework: Week 3, Case Study 1

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

### Exercise 1

A cipher is a secret code for a language. In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of a message to another letter in the alphabet located a fixed distance from the original letter. If our encryption key were `1`, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on. If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

Over the next five exercises, we will create our own Caesar cipher, as well as a message decoder for this cipher. In this exercise, we will define the alphabet used in the cipher.

#### Instructions
- The `string` library has been imported. Create a string called `alphabet` consisting of the space character `' '` followed by (concatenated with) the lowercase letters. Note that we're only using the lowercase letters in this exercise.
"""

import string
# write your code here!
alphabet = " " + string.ascii_lowercase

alphabet

"""### Exercise 2 

In this exercise, we will define a dictionary that specifies the index of each character in `alphabet`.

#### Instructions 
- `alphabet` has already defined in the last exercise. Create a dictionary with keys consisting of the characters in alphabet and values consisting of the numbers from 0 to 26.
- Store this as `positions`.
"""

# write your code here!
positions = {}
p = 0
for c in alphabet:
  positions[c] = p
  p += 1

positions['n']

"""### Exercise 3

In this exercise, we will encode a message with a Caesar cipher.

#### Instructions 

- `alphabet` and `positions` have already been defined in previous exercises. Use `positions` to create an encoded message based on message where each character in message has been shifted forward by 1 position, as defined by positions.
- **Note that you can ensure the result remains within 0-26 using result % 27**
- Store this as `encoded_message`.
"""

message = "hi my name is caesar"
# write your code here!
encoded_message = ''
for c in message:
  for key, values in positions.items():
    if values == (positions[c] + 1) % 27:
      encoded_message += key

encoded_message

"""### Exercise 4

In this exercise, we will define a function that encodes a message with any given encryption key.

#### Instructions 
- `alphabet`, `position` and `message` remain defined from previous exercises. Define a function `encoding` that takes a message as input as well as an int encryption key `key` to encode a message with the Caesar cipher by shifting each letter in message by key positions.
- Your function should return a string consisting of these encoded letters.
- Use `encoding` to encode message using `key = 3` and save the result as `encoded_message`.
Print `encoded_message`.
"""

# write your code here
def encoding(message, encription_key):
  encoded_message = ''
  for c in message:
    for key, values in positions.items():
      if values == (positions[c] + encription_key) % 27:
        encoded_message += key
  return encoded_message

encoded_message = encoding(message, 3)
encoded_message

"""### Exercise 5

In this exercise, we will decode an encoded message.

#### Instructions 
- Use `encoding` to decode `encoded_message`.
- Store your encoded message as `decoded_message`.
- Print `decoded_message`. Does this recover your original message?
"""

# write your code here!
def decoding(encoded_message, decription_key):
  decoded_message = ''
  for c in encoded_message:
    for key, values in positions.items():
      if values == (positions[c] - decription_key) % 27:
        decoded_message += key
  return decoded_message



decoding(encoded_message, 3)