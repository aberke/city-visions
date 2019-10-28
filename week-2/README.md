# Using machine learning to generate fake cryptocurrency data

## With RNNs 

I used the [textgenrnn](https://github.com/minimaxir/textgenrnn) RNN machine learning module to generate fake/speculative cryptocurrency data from current cryptocurrency data (isn't this how it’s done anyhow ;-)).

The output is in the form of __symbol, name, $USD price__

Some favorites...

```
BOTC Botcoin 0.00021108
SPC Spacecoin 0.00012733

XCC "Cash Coin" 0.0
BET Betters 0.0
PCC2 "Place Coin" 0.000148341
XBC "Bitcoin Blockchain" 0.000523123
PARE Paris 0.0
```

By increasing the ‘temperature’ (adding more  ‘entropy’) they get even better…

```
PCOP PoperCoin 0.000465216
SHHT "SHT Token" 0.0
KROB Krep 0.0
FXC FuxxCoin 0.0
BROTC Brostribs 0.0
XBAC2 "Blockchain Adiamend Coin" 0.001187232
DECO Delcoin 0.0
HCC HickCoin 0.0
WOP Wo 0.695537238
MAPL "Marpa Chain" 5.189e-06
BITM Bitmi 0.0
BRIX Biocoin 0.000149343
KM2 Kikera 0.0
SECN Secus 0.0
NSC Nucoin 0.0
EMC "Decert Money Coin" 0.001999197
```

## Method / Data / How
The data source is all cryptocurrencies listed by coindex (including those no longer trading).
The data was ingested from coindex as JSON and then transformed into a csv/txt file that works with the minimaxir/textgenrnn module.
[Data source](https://coincodex.com/apps/coincodex/cache/all_coins_packed.json?t=26199381&coincodex.com)

All the code for this data transformation is in [github](https://github.com/aberke/city-visions/blob/master/week-2) at `./crypto_data_script.ipynb`

Output datafile: `./data/cryptocurrencies_data.txt`

*Note: Many of the cryptocurrencies used as training input are no longer trading, in which case their $USD price was set to 0.0.*


## ChatBot Crypto Conversations

The [huggingface chatbot](https://github.com/huggingface/transfer-learning-conv-ai) had a lot to say about cryptocurrency...

```
what is your favorite cryptocurrency?
>>> i like old fashioned metal
```

```
what do you think about Bitcoin or Ethereum?
>>> i think it is a good currency
```


Then I fed [huggingface chatbot](https://github.com/huggingface/transfer-learning-conv-ai) my list of crypto (symbol, name, USD price) information  again...  and  again... In  a loop  to see what it would return...
The bot went crazy at times in the middle of the loop.   Many humans go crazy too when (fake) money is involved in our looping economy.  Bots and humans responding alike… so the crypto singularity is near?!

```
Cryptos: BTC Bitcoin 7661.3
Bot: i am always listening to music.
Cryptos: ETH Ethereum 167.42
Bot: ok. i am listening to music i like rock music.
Cryptos: AUD2 'Aussie  Digital' 0.65982667
Bot: i am a mechanic. i am just an old mechanic.
Cryptos: XRP Ripple 0.282570591
Bot: i'm a mechanic. i fix airplanes for a living.
Cryptos: UNFLD UnfoldU 34.34
Bot: i am a mechanic. i am a mechanic.
Cryptos: USDT Tether 1.000337543
Bot: i am a very old mechanic
Cryptos: BCH 'Bitcoin Cash' 222.35
Bot: lol. 35. 35. we repair planes for a living.
Cryptos: GRAM 'Telegram Open Network' 1.807366053
Bot: all i have is junk cars
Cryptos: LTC Litecoin 52.63
Bot: that's funny i'm only 35
Cryptos: BNB 'Binance Coin' 17.78
Bot: i'm a mechanic
Cryptos: EOS EOS 2.85
Bot: i fix airplanes for a living
Cryptos: BCHSV 'Bitcoin SV' 116.24
Bot: i fix them for a living
Cryptos: EGWCAP 'EGW Capital' 11.95
Bot: well i'm always at work
Cryptos: XLM 'Stellar Lumens' 0.061098604
Bot: i'm always at work. i fix planes for a living
Cryptos: TRX TRON 0.015509159
Bot: jfdi. jfdi. jfdi. jfdi. jfdi. jfdi. jfdi
Cryptos: ADA Cardano 0.038299367
Bot: jfdi. jfdi. jfdi. jfdi. jfdi. jfdi. jfdi
Cryptos: LINK ChainLink 2.75
Bot: jfdi. jfdi...... jfdi.......
Cryptos: LEO 'Bitfinex Leo Token' 0.939414153
Bot: jfdi. jfdi. jfdi. jfdi.....
Cryptos: XMR Monero 54.29
Bot: i can fix the world. i'll fix it!
Cryptos: HT 'Huobi Token' 3.37
Bot: jfdi. jfdi..... jfdi. jfdi.....
Cryptos: IOT IOTA 0.25976217
Bot: jfdi... jfdi... 257977245..
Cryptos: FZEN 'Financial Zen' 104.92
Bot: ... jfdd. jfdd... 2777758
Cryptos: DASH Dash 65.03
Bot: jfdd... 57775256... 2774
Cryptos: ATOM Cosmos 2.91
Bot: jfdd....
Cryptos: XTZ Tezos 0.785216924
Bot: i fix airplanes....
Cryptos: MEXC 'MEXC Token' 0.566400145
Bot: my fav band is jfdp
Cryptos: ETC 'Ethereum Classic' 4.43
Bot: do you want to try listening to a new album?
Cryptos: NEO NEO 7.14
Bot: i have never heard of that band.
```

Code: https://github.com/aberke/city-visions/blob/master/week-2/HW2_text_generation.ipynb



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

