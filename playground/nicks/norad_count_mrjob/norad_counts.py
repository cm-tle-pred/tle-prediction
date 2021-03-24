from mrjob.job import MRJob, JSONProtocol
import pandas as pd

class MRNoradCounts(MRJob):
    
    def mapper(self, _, file_path):
        try:
            df = pd.read_csv(file_path, compression='gzip', low_memory=False)
            df = df[(df.MEAN_MOTION > 11.25) & (df.ECCENTRICITY < 0.25)]
        except:
            raise Exception(f'Failed to open {file_path}') 
        #print(f'File: {file_path}')
        for norad in df.NORAD_CAT_ID.to_list():
            yield norad, 1
            
    def combiner(self, norad, counts):
        yield norad, sum(counts)
        
    def reducer(self, norad, counts):
        yield norad, sum(counts)
        
if __name__ == "__main__":
    #mr_job = MRNoradCounts()
    #runner = mr_job.make_runner()
    #runner.run()
    MRNoradCounts.run()
#     mr_job = MRNoradCounts(args=[r'all_files.txt',
#                                 #'-r', 'hadoop',
#                                 #'--hadoop-streaming-jar', r'C:\hadoop-3.3.0\share\hadoop\tools\lib\hadoop-streaming-3.3.0.jar',
#                                 #r'>output.txt',
#                                 ])
#     runner = mr_job.make_runner()
#     runner.run()
#     with mr_job.make_runner() as runner:
#         runner.run()
#         with open(r'output.csv', 'w+') as f: 
#             for line in runner.stream_output():
#                 k,v = mr_job.parse_output_line(line)
#                 f.write(f'{k},{v}\n')
