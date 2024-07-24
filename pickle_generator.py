import os
import json
import pickle

data_path = "data/"
speech_path = "data/LibriSpeech-wav/test-clean/"
AV_path = "synthetic_dataset/v16/test/"

pickle_name ="test.pickle"
folder_list = os.listdir(data_path+AV_path)
unique_id = 0
path_list=[]

for folders in folder_list:
	folder_path  = data_path + AV_path + folders +"/"

	json_path = folder_path+"cleaned_metadata_v3.json"

	f = open(json_path)

	data = json.load(f)

	len_data = len(data)

	for i in range(len_data):
		# print("i ",i,"  len data",len_data)
		full_dict = data[i]
		speakers = full_dict['speakers']
		num_speakers=len(speakers)
		for j in range(num_speakers):
			speaker_id = str(speakers[j]['id']) #needed
			speaker_location = speakers[j]['location'] #needed
			speaker_gender = speakers[j]['gender'] #needed
			speaker_speech = speakers[j]['speech']
			split_speech = speaker_speech.split("-")
			full_clean_speech_path = speech_path + split_speech[0] + "/" + split_speech[1] + "/" + speaker_speech +".wav" #needed

			view_points = full_dict['viewpoints']
			num_view_points = len(view_points)
			unique_id = unique_id+1
			for k in range(num_view_points):
				view_location = view_points[k]['location']  #needed
				view_image = view_points[k]['image']
				mono_rir_path = view_points[k]['mono_rir'][speaker_id]
				full_mono_rir_path = mono_rir_path
				full_reverb_speech_path = full_mono_rir_path.replace(".wav","_reverb_speech.wav") #needed

				dict_data={}
				dict_data['unique_id'] = unique_id
				dict_data['speaker_id'] = speaker_id
				dict_data['speaker_location'] = speaker_location
				dict_data['speaker_gender'] = speaker_gender
				dict_data['clean_speech_path'] = full_clean_speech_path
				dict_data['view_location'] = view_location
				dict_data['view_image'] = view_image
				dict_data['mono_rir_path'] = full_mono_rir_path
				dict_data['reverb_speech_path'] = full_reverb_speech_path
				path_list.append(dict_data)



with open(pickle_name, 'wb') as f:
	pickle.dump(path_list, f, protocol=2)


data_path = "data/"
speech_path = "data/LibriSpeech-wav/train-clean-360/"
AV_path = "synthetic_dataset/v16/train/"

pickle_name ="train.pickle"
folder_list = os.listdir(data_path+AV_path)
unique_id = 0
path_list=[]

for folders in folder_list:
	folder_path  = data_path + AV_path + folders +"/"

	json_path = folder_path+"cleaned_metadata_v3.json"

	f = open(json_path)

	data = json.load(f)

	len_data = len(data)

	for i in range(len_data):
		# print("i ",i,"  len data",len_data)
		full_dict = data[i]
		speakers = full_dict['speakers']
		num_speakers=len(speakers)
		for j in range(num_speakers):
			speaker_id = str(speakers[j]['id']) #needed
			speaker_location = speakers[j]['location'] #needed
			speaker_gender = speakers[j]['gender'] #needed
			speaker_speech = speakers[j]['speech']
			split_speech = speaker_speech.split("-")
			full_clean_speech_path = speech_path + split_speech[0] + "/" + split_speech[1] + "/" + speaker_speech +".wav" #needed

			view_points = full_dict['viewpoints']
			num_view_points = len(view_points)
			unique_id = unique_id+1
			for k in range(num_view_points):
				view_location = view_points[k]['location']  #needed
				view_image = view_points[k]['image']
				mono_rir_path = view_points[k]['mono_rir'][speaker_id]
				full_mono_rir_path = mono_rir_path
				full_reverb_speech_path = full_mono_rir_path.replace(".wav","_reverb_speech.wav") #needed

				dict_data={}
				dict_data['unique_id'] = unique_id
				dict_data['speaker_id'] = speaker_id
				dict_data['speaker_location'] = speaker_location
				dict_data['speaker_gender'] = speaker_gender
				dict_data['clean_speech_path'] = full_clean_speech_path
				dict_data['view_location'] = view_location
				dict_data['view_image'] = view_image
				dict_data['mono_rir_path'] = full_mono_rir_path
				dict_data['reverb_speech_path'] = full_reverb_speech_path
				path_list.append(dict_data)

with open(pickle_name, 'wb') as f:
	pickle.dump(path_list, f, protocol=2)

data_path = "data/"
speech_path = "data/LibriSpeech-wav/dev-clean/"
AV_path = "synthetic_dataset/v16/val/"

pickle_name ="val.pickle"
folder_list = os.listdir(data_path+AV_path)
unique_id = 0
path_list=[]

for folders in folder_list:
	folder_path  = data_path + AV_path + folders +"/"

	json_path = folder_path+"cleaned_metadata_v3.json"

	f = open(json_path)

	data = json.load(f)

	len_data = len(data)

	for i in range(len_data):
		# print("i ",i,"  len data",len_data)
		full_dict = data[i]
		speakers = full_dict['speakers']
		num_speakers=len(speakers)
		for j in range(num_speakers):
			speaker_id = str(speakers[j]['id']) #needed
			speaker_location = speakers[j]['location'] #needed
			speaker_gender = speakers[j]['gender'] #needed
			speaker_speech = speakers[j]['speech']
			split_speech = speaker_speech.split("-")
			full_clean_speech_path = speech_path + split_speech[0] + "/" + split_speech[1] + "/" + speaker_speech +".wav" #needed

			view_points = full_dict['viewpoints']
			num_view_points = len(view_points)
			unique_id = unique_id+1
			for k in range(num_view_points):
				view_location = view_points[k]['location']  #needed
				view_image = view_points[k]['image']
				mono_rir_path = view_points[k]['mono_rir'][speaker_id]
				full_mono_rir_path = mono_rir_path
				full_reverb_speech_path = full_mono_rir_path.replace(".wav","_reverb_speech.wav") #needed

				dict_data={}
				dict_data['unique_id'] = unique_id
				dict_data['speaker_id'] = speaker_id
				dict_data['speaker_location'] = speaker_location
				dict_data['speaker_gender'] = speaker_gender
				dict_data['clean_speech_path'] = full_clean_speech_path
				dict_data['view_location'] = view_location
				dict_data['view_image'] = view_image
				dict_data['mono_rir_path'] = full_mono_rir_path
				dict_data['reverb_speech_path'] = full_reverb_speech_path
				path_list.append(dict_data)

with open(pickle_name, 'wb') as f:
	pickle.dump(path_list, f, protocol=2)