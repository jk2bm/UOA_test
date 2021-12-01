import pandas as pd
import sys
import math
from datetime import datetime

data = pd.read_csv("")
pd.options.display.max_columns=2000
pd.options.display.max_rows=20000
data.sort_values(["Vol/OI","Volume"], axis=0, ascending=[False, False], inplace=True)
data.drop(data.tail(1).index,inplace=True)
df = pd.DataFrame(data, columns = ["Symbol", "Type", "Price", "Strike", "Exp Date", "DTE", "Midpoint", "Volume", "Open Int", "IV"])

## define Cox_Ross_Rubinstein binomial model
def Cox_Ross_Rubinstein_Tree (S,K,T,r,sigma,N, Option_type):
    
    # Underlying price (per share): S; 
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Number of binomial steps: N;

        # The factor by which the price rises (assuming it rises) = u ;
        # The factor by which the price falls (assuming it falls) = d ;
        # The probability of a price rise = pr ;
        # The probability of a price fall = pf ;
        # discount rate = disc ;
    
    u=math.exp(sigma*math.sqrt(T/N));
    d=math.exp(-sigma*math.sqrt(T/N));
    pr=((math.exp(r*T/N))-d)/(u-d);
    pf=1-pr;
    disc=math.exp(-r*T/N);

    St = [0] * (N+1)
    C = [0] * (N+1)
    
    St[0]=S*d**N;
    
    for j in range(1, N+1): 
        St[j] = St[j-1] * u/d;
    
    for j in range(1, N+1):
        if Option_type == 'Put':
            C[j] = max(K-St[j],0);
        elif Option_type == 'Call':
            C[j] = max(St[j]-K,0);
    
    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc*(pr*C[j+1]+pf*C[j]);
            
    return C[0]


## define Jarrow_Rudd binomial model    
def Jarrow_Rudd_Tree (S,K,T,r,sigma,N, Option_type):

    # Underlying price (per share): S; 
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r;
    # Volatility: sigma;
    # Steps: N;
    
        # The factor by which the price rises (assuming it rises) = u ;
        # The factor by which the price falls (assuming it falls) = d ;
        # The probability of a price rise = pr ;
        # The probability of a price fall = pf ;
        # discount rate = disc ;
        
    u=math.exp((r-(sigma**2/2))*T/N+sigma*math.sqrt(T/N));
    d=math.exp((r-(sigma**2/2))*T/N-sigma*math.sqrt(T/N));
    pr=0.5;
    pf=1-pr;
    disc=math.exp(-r*T/N);

    St = [0] * (N+1)
    C = [0] * (N+1)
    
    St[0]=S*d**N;
    
    for j in range(1, N+1): 
        St[j] = St[j-1] * u/d;
    
    for j in range(1, N+1):
        if Option_type == 'Put':
            C[j] = max(K-St[j],0);
        elif Option_type == 'Call':
            C[j] = max(St[j]-K,0);
    
    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc*(pr*C[j+1]+pf*C[j]);
            
    return C[0]

def identifyExtraUnusual(df):
  extraUnusualList = []
  expectedMoves = []
  for n in range(0, len(df)):
    strike = float(df.at[n, "Strike"])
    price = float(df.at[n, "Price"])
    midpoint = float(df.at[n, "Midpoint"])
    iv = df.at[n, "IV"]
    vol = float(iv.replace("%",""))/100
    dte = df.at[n, "DTE"]
    breakEven = strike + midpoint
    expectedMove = abs(price * vol * math.sqrt(dte/365))

    if (dte >= 365):
      pass
    elif (expectedMove / price >= .5):
      extraUnusualList.append(df.at[n, "Symbol"])
    elif (expectedMove / price >= 0.25):
      if (dte <= 7):
        extraUnusualList.append(df.at[n, "Symbol"])
    elif (expectedMove / price >= 0.125):
      if (dte <= 2):
        extraUnusualList.append(df.at[n, "Symbol"])
  return extraUnusualList

unusualList = []
verdicts = []
extraUnusualList = identifyExtraUnusual(df)
for n in range(0, 10):
  unusualList.append(df.at[n, "Symbol"])

unusualList = unusualList + extraUnusualList
unusualList = list(dict.fromkeys(unusualList))

def estimateRelValues(df):
  relc = []
  relp = []
  dfc = df.loc[df["Type"] == "Call"].reset_index()
  dfp = df.loc[df["Type"] == "Put"].reset_index()
  for n in range(0, len(dfc)):
    relc.append(float(dfc.at[n, "Midpoint"]*dfc.at[n, "Volume"]))
  for n in range(0, len(dfp)):
    relp.append(float(dfp.at[n, "Midpoint"]*dfp.loc[n, "Volume"]))
  sumc = round(sum(relc),2)
  sump = round(sum(relp),2)
  if sump == 0:
    verdict = "Extremely Bullish"
  elif sumc == 0:
    verdict = "Extremely Bearish"
  elif sumc > sump and sumc / sump >= 1.5:
    pwr = str(round((sumc/sump*100)-100,2))+ "%"
    verdict = str(pwr)+" Bullish"
  elif sump > sumc and sump / sumc >= 1.5:
    pwr = str(round((sump/sumc*100)-100,2))+ "%"
    verdict = pwr+" Bearish"
  else:
    verdict = "Neutral"
  return verdict

dfc = pd.DataFrame()

for n in range(0, len(unusualList)):

  contract = df.loc[df["Symbol"] == unusualList[n]]
  print(unusualList[n]+ " " + estimateRelValues(contract) + "\n")
  contract = contract.reset_index()
  costavg = []
  for i in range(0, len(contract)):
    vol = float(contract.at[i, "IV"].replace("%",""))/100
    if vol != 0:
      try:
        cost1 = Cox_Ross_Rubinstein_Tree(contract.at[i, "Price"], contract.at[i, "Strike"], contract.at[i, "DTE"]/365, 0, vol, 1, contract.at[i, "Type"])
        cost2 = Jarrow_Rudd_Tree(contract.at[i, "Price"], contract.at[i, "Strike"], contract.at[i, "DTE"]/365, 0, vol, 1, contract.at[i, "Type"])
        costavg.append(round((cost1 + cost2)/2,2))
      except:
        costavg.append(0)
    else:
      costavg.append(0)
  
  expectedMoves = []
  for n in range(0, len(contract)):
    price = float(contract.at[n, "Price"])
    iv = contract.at[n, "IV"]
    vol = float(iv.replace("%",""))/100
    dte = contract.at[n, "DTE"]
    expectedMove = round(abs(price * vol * math.sqrt(dte/365)),2)
    expectedMoves.append(expectedMove)

  contract.insert(8, "Est Value", costavg)
  contract.insert(9, "Exp Move", expectedMoves)
  print(contract.to_string(index = False))
  contract.at[0, "Sentiment"] = estimateRelValues(contract)
  dfc = dfc.append(contract)
  print("\n")
  #verdicts.append(unusualList[n]+ " " + estimateRelValues(symbol))

date = str(datetime.now().date())
filename = "UOA_Analyzed " + date + ".csv"
dfc.to_csv(filename, index = False)
#for item in verdicts:
#  print(item)
