import os
import openai
import json
import pdb
from tqdm import tqdm

# openai.api_key = ""
json_name = "eurosat.json"

category_list = [ "Annual Crop Land", "Forest" , "Herbaceous Vegetation Land", "Highway or Road", "Industrial Buildings", "Pasture Land", "Permanent Crop Land", "Residential Buildings", "River", "Sea or Lake"]
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U']

for category in tqdm(category_list):

	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = []
	prompts.append("Describe a satellite photo of " + article + " " + category)
	prompts.append("Describe " + article + " " + category + " a as it would appear in an aerial image")
	prompts.append("How does " + article + " " + category + " look like in an satellite photo?")
	prompts.append("How can you identify " + article + " " + category + "  in an aerial photo?")
	prompts.append("Describe the satellite photo of " + article + " "  + category)
	prompts.append("Describe an aerial photo of "  + article + " "  + category)
	# prompts.append("Describe a medical image of "  + article + " "  + category )

	all_result = []
	for curr_prompt in prompts:
		response = openai.Completion.create(
		    engine="gpt-3.5-turbo-instruct",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 50,
			n=10,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)