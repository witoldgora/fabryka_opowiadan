Project Name: Fabryka Opowianań

This project strives to buid a fully function story or book writing assistent with pipelined flow and control options at different stages.
This concept depicted in fabryka_opowiadan.png extends the original task described in 32__app_ebooks_short.ipynb with the following features:

- generation of audio version of the book in MP3 format
- HTML version of the final book version with text and illustrations
- controls of AI parameters at various stages of the content generation process
- store/restore fo all intermediate stages of the content generation process
- multiiple choices of content version for intermediate stages
- optional user input from a .TXT file uploaded from a local storage


The text processing assumes extraction of the key points like: characters, places, story plots etc. that are subject for futher processing.
The variations of the story variants or adherence to the specified inputs is controlled with the LLM settings e.g. Temperature parameter. 