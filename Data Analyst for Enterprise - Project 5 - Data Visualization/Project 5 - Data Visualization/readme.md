# Flight Data Exploration
## by Leslie Rodriguez


## Dataset

The flight dataset I will be using in this project reports flights in the United States, including carriers, arrival and departure delays, and reasons for delays, from 2008. The data comes originally from RITA where it is described in detail here - https://www.transtats.bts.gov/Fields.asp?Table_ID=236.

Here are the variable descriptions for my data:
Variable descriptions
Name Description
1 Year 1987-2008
2 Month 1-12
3 DayofMonth 1-31
4 DayOfWeek 1 (Monday) - 7 (Sunday)
5 DepTime actual departure time (local, hhmm)
6 CRSDepTime scheduled departure time (local, hhmm)
7 ArrTime actual arrival time (local, hhmm)
8 CRSArrTime scheduled arrival time (local, hhmm)
9 UniqueCarrier unique carrier code
10 FlightNum flight number
11 TailNum plane tail number
12 ActualElapsedTime in minutes
13 CRSElapsedTime in minutes
14 AirTime in minutes
15 ArrDelay arrival delay, in minutes
16 DepDelay departure delay, in minutes
17 Origin origin IATA airport code
18 Dest destination IATA airport code
19 Distance in miles
20 TaxiIn taxi in time, in minutes
21 TaxiOut taxi out time in minutes
22 Cancelled was the flight cancelled?
23 CancellationCode reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
24 Diverted 1 = yes, 0 = no
25 CarrierDelay in minutes
26 WeatherDelay in minutes
27 NASDelay in minutes
28 SecurityDelay in minutes
29 LateAircraftDelay in minutes


## Summary of Findings

There are over 7 million flights in the dataset with close to 30 features. For my exploration I only leverage a 1 million datapoint sample.
ATL, ORD, and DFW are the top 3 Origin airports, and ATL, ORD, DFW are the top 3 Destination airports.
A small percentage of the 2008 flight dataset experience cancellations, about less than .05 percent. 
The top 2 reasons for cancellation are Carrier Delay and Weather Delay.
Some trends I've found in the data is that Arrival Delay has many negative values throughout the dataset, meaning that the flights tend to arrive sooner than expected.
Most delays overall tend to be within 1 hour, but there are several extreme cases of delays that are over 300 minutes (5+ hours).
There was obvious relationships in arrival times and distance, with high distance flights requiring more time.
Flights and delays varies throughout the year, where we may see some more delays during winter season, and even cancellations.

## Key Insights for Presentation

Most of my insights are around overall time in minutes spent on flights, and seeing their different impacts based on airports the flights are flying into/flying from, and overall trends through the 2008 year. Although the top airports see a variety of delays, most delays are experienced for shorter flights rather then the long flights. Some airports are also less affected by distance and experience delays for external reasons or potentially weather reasons.

## Resources used
Some resources that were used for this project include google search results, examples given in classroom topics and previous samples provided by the courses.