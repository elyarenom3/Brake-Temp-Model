## Overview + Idea Behind Model 

This repo contains Python code to predict left and right brake temps in a F1 car, using data from six races. I really really enjoyed building this and I would really love some feedback on how I can improve :) . I decided to use an ML gradient boosting machine model because in the context of F1 where you constantly have new race data, it can easily be added to the model periodically, improving accuracy over time, plus gradient boosting machines are great for predicting complex, non-linear relationships between features and target variables. I decided to go with XGBoost, mainly because it includes L1 and L2 regularization, which helps prevent the model from overfitting.

Before creating my model, I did some correlation and regression testing to confirm some physics/common-sense based theories I had about how the channels would interact (eg. I made an acceleration feauture because when the car accelerates theres more air going into the ducts so the brakes cool down faster). I realised that only taking the negative parts of my acceleration to model braking periods was not enough, because the car still decelerates when you lift the throttle without even touching the brake, so I ran some tests to see at what threshold a negative delta speed corresponds to actual braking and found -2.

From there, I realised through splitting the data set and running correlation calculations that interaction between our variables is completely different in braking periods and non-braking periods so I decided to model delta brake temperature for the left and right brakes seperately for braking and non-braking periods. I modelled delta brake temp instead of normal brake temp since that what showed strongest correlations with other variables, and then in the model_testing.py file, I checked whether we’re braking (delta speed of less than 2) or not, and then applied the model (which I create using xgboost in the model_testing.py file) accordingly. Since my model calculates the delta, I simply add it on to the last known value and then set it to be the newest value.

The main problem is that the braking parts (when delta speed is below 2) of the prediction are way more accurate than the rest, so if I had more time I would do some more feature engineering and experiment with model complexity like depth of trees to try and understand the how different factors interact to impact the temperature of brakes when they’re not braking. 


### Improvements

I spent around three hours on this, as was suggested in the document, but if I had more time I would:

1. **Create a lap identifier** by looking at the track and then matching up its turns to left and right braking data, as the left tyre heats more when making a right turn and vice versa so I could probably identify a lap by looking at a given pattern of left and right turns repeating. I tried using peak detection, clustering and a Fourier transform to identify repeating patterns but it was inconclusive.


2. **Do more feature engineering** like for the duration of braking periods, rolling averages of temperatures and speeds, and binary indicators for specific race conditions (e.g., pit stops, rain). That could improve prediction accuracy.


3. **More detailed error analysis and testing**, identifying patterns in prediction errors and correlating them with specific race conditions or driving behaviors could help in refining my models further.

### Initial Data Requirement

The model uses both direct measurements and engineered features that capture temporal changes. Enough data is needed to allow the model to learn the significance of these features across different racing conditions. Basically, having data from several races, covering a variety of conditions (e.g., different tracks, weather conditions), would provide a solid foundation for training it.

### Forecasting Capability

With more time, I would do a deeper dive on performance metrics (e.g., RMSE, MAE) as the forecast horizon extends to know when results are no longer trustworthy. GBM models like XGBoost aren't really designed for multi-step time series forecasting because they predict the next step based on past and current data so the further into the future you try to forecast, the more uncertainty accumulates. To mitigate this, I would create features based on rolling windows (e.g., the average of the last n observations) to capture more of the temporal context in the features.

- **Unseen Tracks**: Because my model is trained mostly on the underlying principles of brake heating and cooling, rather than memorizing track-specific patterns, it should respond relativelt well to a completely unseen track. Also, since the training data includes a diverse set of tracks and conditions, its even more likely to generalize well to new tracks. If the new tracks are significantly different from those in the training set (e.g., much longer straights, different surface materials), the model may struggle without fine-tuning or retraining on data that includes similar conditions to the new track.

### Most important factors: (used built-in xgboost importance)

* Most Important Feature for Model R Greater Than or Equal to 2: TTrack with an importance of 0.48393726
* Most Important Feature for Model L Greater Than or Equal to 2: vCar with an importance of 0.5056588
* Most Important Feature for Model R Less Than -2: deltaSpeed with an importance of 0.9942036
* Most Important Feature for Model L Less Than -2: deltaSpeed with an importance of 0.9936026

tiny note: I’m assuming that the few negative values for speed, because they’re so small, are a result of how the sensor or the data logging system handles errors or underflows. To address this, if given more time, I would do more thorough preprocessing + cleaning of the data and potentially consult documentation for the data logging system to 
understand how an anomaly might occur.
