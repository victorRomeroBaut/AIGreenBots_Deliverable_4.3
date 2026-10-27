# predictiveNav
PredictiveNav component introduces a World Model based on the Active Inference mathematical model. 

## Configuration parameters
As any other component, *predictiveNav* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
Proxies.CameraRGBDSimple = "camerargbdsimple:tcp -h localhost -p 10096"
Proxies.OmniRobot        = "omnirobot:tcp -h localhost -p 10004"
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <predictiveNav's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/predictiveNav config
```
