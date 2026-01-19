### Fast or Secure? Push the Limit of Privacy Leakage Threat via Charging Side-Channel Attacks

##### Models Overview

Our models target duplex audio/screen privacy inference in a charging side-channel attack setting, using synchronized two-channel VBUS (voltage) and IBUS (current) traces.

###### 3-Layer CNN:

Used for output operation inference (e.g., recognizing screen display content, loudspeaker eavesdropping). It processes synchronized voltage and current traces from charging side channels to classify device output activities. For screen display tasks specifically, the segmented signals are zero-padded to a fixed length of 3000 samples to facilitate subsequent network processing; for all other tasks (e.g., loudspeaker eavesdropping), the signal is zero-padded to 1000 samples.

###### Attention-Based ResNet:

Used for input operation inference (e.g., password/keystroke inference, microphone eavesdropping). It extracts dynamic features from two-channel (voltage + current) power traces to identify user input behaviors.



