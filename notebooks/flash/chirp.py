import numpy as np
import pandas as pd
from functools import reduce

def get_chirp_subevents(sync_path, start_end_path, repeated_path, output_file, names, times):
	sync = pd.read_csv(sync_path)
	start_end = np.loadtxt(start_end_path, dtype=int)
	start_frame, end_frame = start_end[:, 0], start_end[:, 1]
	repetead = np.loadtxt(repeated_path, dtype=int)

	protocol_name = 'chirp'  # name of protocol in event_list file

	chirp_filter = sync['protocol_name'] == protocol_name

	# extra_description filter

	chirp_times = sync[chirp_filter]

	if chirp_times.shape[0] == 0:
		return None

	sr = 20000.0
	fps = 60

	start_trans = np.array([[0] + times[:-1]]).cumsum() * fps
	end_trans = np.array(times).cumsum() * fps - 1

	events_chirp = []
	for _, kchirp in enumerate(chirp_times.itertuples()):
		rep_trans = np.zeros_like(times)
		sub_set = np.logical_and(start_frame >= kchirp.start_event, 
								 end_frame <= kchirp.end_event)
		
		sub_start_frame = start_frame[sub_set]
		sub_end_frame = end_frame[sub_set]
		
		for ktrans, (start, end) in enumerate(zip(start_trans, end_trans)):
			rep_trans[ktrans] = np.logical_and(
				repetead >= sub_start_frame[start + rep_trans[:ktrans].sum()],
				repetead <= sub_end_frame[end + rep_trans[:ktrans].sum()]
			).sum()
				
		idx = np.where(start_frame == kchirp.start_event)[0][0]
		start_event = start_frame[idx + start_trans + rep_trans]
		end_event = end_frame[idx + end_trans + rep_trans]
		
		df = pd.DataFrame(
			{
				'n_frames': end_trans - start_trans + rep_trans + 1,
				'start_event': start_event,
				'end_event': end_event,
				'start_next_event': end_event,
				'event_duration': end_event - start_event,
				'event_duration_seg': (end_event - start_event) / sr,
				'inter_event_duration': 0,
				'inter_event_duration_seg': 0.0,
				'protocol_name': protocol_name,
				'extra_description': names,
				'protocol_spec': kchirp.extra_description,
				'rep_name': kchirp.repetition_name,
				'repetead_frames': '',
				'#repetead_frames': rep_trans,
			})
		events_chirp.append(df)
	events = reduce(lambda x, y: pd.concat([x, y]), events_chirp)
	events.to_csv(output_file, sep=';', index=False)
	return events