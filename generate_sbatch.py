#!/usr/bin/python

import yaml
import argparse
import numpy as np

class ScriptGenerator:
	def __init__(self, args):
		self.config_filename = args.config_filename
		with open(self.config_filename, 'r') as f:
			self.config = yaml.safe_load(f)	

		with open(self.config['template_file'], 'r') as f:
			self.template = f.read()
		self.slurm_config = self.config['slurm']
		self.job_config = self.config['job_params']

	def _get_job_params(self):
		num_rotations = self.job_config['num_rotations']
		params = ""
		params += "\t--num_parallel_calls 16 \\\n"
		params += "\t--log_dir %s \\\n" % self.job_config['log_dir']
		params += "\t--batch_size %d \\\n" % self.job_config['batch_size']
		
		dataset_dir = self.job_config['dataset_dir'] % num_rotations
		params += "\t--dataset_dir %s \\\n" % dataset_dir
		
		out_name = "%s_%s_rot%d" % (self.job_config['modality'],
					    self.job_config['loss'], 
					    num_rotations)
		if "learning_rate" in self.job_config:
			out_name += "_%s" % str(self.job_config['learning_rate'])
		if "aug_ratio" in self.job_config:
			out_name += "_augratio%s" \
				    % ("%.2f" % self.job_config['aug_ratio']).split('.')[1]
		params += "\t--out_name %s \\\n" % out_name

		return params, out_name		

	def run(self):
		# Job
		params, out_name = self._get_job_params()
		self.template = self.template.replace('[JOB_PARAMS]', params)

		# Slurm
		excl_compute = self.slurm_config['exclude_compute']
		self.template = self.template.replace('[EXCLUDE_COMPUTE]', excl_compute)

		log_file = self.slurm_config['log_file'] 
		log_file = log_file % out_name
		self.template = self.template.replace('[LOGFILE]', log_file)
 
		cuda_str = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$%d" \
			   if "gpu" in self.slurm_config else ""
		self.template = self.template.replace('[CUDADEVICE]', cuda_str)

		singularity_img = self.slurm_config['singularity_img'] \
				  % (self.job_config['modality'], self.job_config['loss'])
		self.template = self.template.replace('[SINGULARITY_IMG]', singularity_img)
		
		singularity_script = self.slurm_config['singularity_script'] \
				     % (self.job_config['modality'])
		self.template = self.template.replace('[SINGULARITY_SCRIPT]', singularity_script)
			
		print(self.template)
		with open("%s.sh" % out_name, 'w') as f:
			f.write(self.template)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate a training script from config')
	parser.add_argument("--config_filename", type=str, default=None, help='Config filename')
	args = parser.parse_args()
	
	sg = ScriptGenerator(args)
	sg.run()
