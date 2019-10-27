# Using machine learning to generate fake cryptocurrency data

## With RNNs 

I used the [textgenrnn](https://github.com/minimaxir/textgenrnn) RNN machine learning module to generate fake/speculative cryptocurrency data from current cryptocurrency data (it’s this how it’s done anyhow ;-)).

The output is in the form of __symbol, name, $USD price__

Some favorites...

```
BOTC Botcoin 0.00021108
CRT "Credit Token" 0.000653993
BOT BootCoin 0.000129398
SPC Spacecoin 0.00012733

XCC "Cash Coin" 0.0
XCC "Chin Coin" 0.000139998
GET2 "Gene Token" 0.000113792
BET Betters 0.0
DARE Darto 0.0
STT3 "Start Coin" 0.0
PCC2 "Place Coin" 0.000148341
XBC "Bitcoin Blockchain" 0.000523123
PARE Paris 0.0
BARE BitcoinA 0.001190911
```

And then by increasing the ‘temperature’ (adding more  ‘entropy’) they get even better…

```
CTA1 "Currenity Token" 0.0
KROB Krep 0.0
FXC FuxxCoin 0.0
EMC "Decert Money Coin" 0.001999197
BROTC Brostribs 0.0
XBAC2 "Blockchain Adiamend Coin" 0.001187232
DECO Delcoin 0.0
HCC HickCoin 0.0
WOP Wo 0.695537238
MAPL "Marpa Chain" 5.189e-06
BITM Bitmi 0.0
BRIX Biocoin 0.000149343
SHHT "SHT Token" 0.0
KM2 Kikera 0.0
SECN Secus 0.0
NSC Nucoin 0.0
PCOP PoperCoin 0.000465216
```

## Method / Data / How
The data source is all cryptocurrencies listed by coindex (including those no longer trading).
The data was ingested from coindex as JSON and then transformed into a csv/txt file that works with the minimaxir/textgenrnn module.
[Data source](https://coincodex.com/apps/coincodex/cache/all_coins_packed.json?t=26199381&coincodex.com)

All the code for this data transformation is in [github](https://github.com/aberke/city-visions/blob/master/week-2) at `./crypto_data_script.ipynb`

Output datafile: `./data/cryptocurrencies_data.txt`

*Note: Many of the cryptocurrencies used as training input are no longer trading, in which case their $USD price was set to 0.0.*




## The Real Data is weird enough...

In the debugging process I encountered names of existing crypto coins that are  weirder than ML could have randomly come up with, and are trading at non-zero $USD values….

```
MAY "Theresa May Coin" 0.000197574
TSE "Tattoocoin (Standard Edition)" 0.000217536
FLUZ 'Fluz Fluz' 0.022218693
LALA 'LALA World' 0.017091302
POE Po.et 0.00214985
EMC2 Einsteinium 0.042217778
FAT Fatcoin 0.018319265
GBC2 'Gold Bits Coin' 0.022218693
```

Full list of my input cryptos in github here: https://github.com/aberke/city-visions/blob/master/week-2/data/cryptocurrencies_data.txt 

This is a classic example of why you should inspect your data before processing your data.  I should have realized the futility of this art project  from the start: I wasn’t going to make something weirder than the world of cryptofans had already invested in.

