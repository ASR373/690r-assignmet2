data_dict = {}
for folder in sorted(os.listdir(data_dir)):
    subject = re.match(r'subject(\d+)', folder)
    if subject is None:
        continue
    subject = int(subject[1])
    
    path = os.path.join(data_dir, folder)
    for file in sorted(os.listdir(path)):
        activity = re.match(r'acc_(\w+)_forearm.csv', file)
        if activity is None:
            continue
        activity = activity[1]
        
        data = pd.read_csv(os.path.join(path, file))
        data = data[['attr_x', 'attr_y', 'attr_z']].to_numpy()
        
        # save
        data_dict[subject, activity] = data
np.savez('data/activity_data.npz', **{'data': data_dict})