# CSI Semantic Segmentation

Goal: To verify that semantic latent representations of CSI-image from a pre-trained model does indeed contain meaningful semantics. In order to achieve the goal, visualize semantic segmentation and semantic latent presentation side by side.

- Step-1: dataset generation (your can use /home/dgx/cpyen/MyProject/ijepa/data_gen/generate_csi_cdlc.py as a reference)
  - Call 3GPP TR38.901 channel mode from Sionna (~/sionna) and generate 80,000 samples of RB-level downlink MISO channel with the following parameters;
    - channel model: CDL-A
    - parameters:
      - CARRIER_FREQUENCY =  7.0e9 (7 GHz)
      - SUBCARRIER_SPACING =  30.0e3 (30 KHz)
      - DELAY_SPREAD  = 100.0e9 (100 ns)
      - UE speed = 30 km/h (8.3 m/s, mobile speed for Doppler)
      - NUM_TX_ANT=32 with num_rows=2, num_cols=8, cross-polarized
      - NUM_RX_ANT=1 
      - N_PRB=48
  - Perform FFT to antenna port domain to convert it to angular domain
    - per-polarization, perform 2D FFT, i.e. $N_{FFT1}=64$ point on $N_2$ (vertical) and $N_{FFT2}=256$ on $N_1$ (horizontal)
  - Arrange the samples in delay-angular domain for each PRB. The tensor shape is $[\text{samples},\text{PRB}, \text{delay}, \text{departure\_angles}]$, where 
    - $\text{departure\_angle}$ is 2 scalars of $(\text{azimuth\_angle\_of\_departure(AoD)},\text{zenith\_angle\_of\_departure(ZoD)})$
  - Record number of path for each sample on a separate file
  - Verify the data samples
  - Visualization
    - pick 3 data sample randomly and for each sample, plot two 2D figures, one is delay-AoD and the other is delay-ZoD, also write down number of paths
  - Split the dataset into a training dataset and a testing dataset which contain 10,000 samples
- Step-2: Training Latent MIM with the training dataset
  - Show the loss function on the terminal every 100 epochs 

- Step-3: Obtain latent representations using the pre-trained Online Encoder
  - feed the testing dataset without masking
  - output patch level representations
- Step-4: unsupervised segmentation
  - perform hierarchical clustering
    - For example, use Agglomerative Hierarchical Clustering (*AgglomerativeClustering* in sklearn.cluster)
    - You can suggest better alternatives if any
  - generate unsupervised segmentation maps
    - For example, up-sample to an image that has the same size as the input by interpolation of the patch level representations
    - You can suggest better alternatives if any
- Step-5: visualization
  - Randomly pick 5 testing data samples
  - draw the unsupervised segmentation maps
  - draw t-SNE of the latent representations, using the same color code as the segmentation map
  - put these 3 plot side by side for comparisons, also write down the number of paths of the channel

