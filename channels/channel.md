## Channel Generation
### Channel generation
The channel is generated using MATLAB and QuaDriGa. The installation and download of QuaDriGa can be found in [here](https://quadriga-channel-model.de/).

The `generate_channel.m` will automatically save the channel file as `channel.mat`. 
To test the generated channel file, change the `channel_path` in the `main.py` as `channels/channel.mat`.

### Acknowledgement
The channel generation file is modified from the implementation in [MMNet](https://github.com/mehrdadkhani/MMNet/tree/master).