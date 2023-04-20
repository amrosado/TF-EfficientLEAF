# TF-EfficientLEAF
## TensorFlow Implementation of EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use
## About
This repository aims to reproduce the Efficient LEAF front-end model using TensorFlow 
and Keras towards enabling learnable audio frontends in TensorFlow without using Gin 
and Lingvo dependencies limiting the usability and compatibility of the original LEAF library. The 
code heavily reproduces the original code featured in the EUSIPCO EfficientLEAF: 
A Faster Learnable Audio Frontend of Questionable Use published by Jan Schlüter 
and Gerald Gutenbrunner (https://arxiv.org/abs/2207.05508).  The original GitHub 
repo can be found at: https://github.com/CPJKU/EfficientLEAF.

Thank you to Jan and Gerald for answering my questions regarding their 
implementation and their support of making Efficient LEAF available 
to the TensorFlow community.

## Tested with:
* Python 3.9
* Tensorflow 2.10.x (Windows) and Tensorflow 2.12.x (Linux)

## Using:
* Nvidia CUDA >=11.2.2
* Nvidia CUDNN >=8.1.0.77

## Dependencies:
* Tensorflow 2
* Huggingface Datasets
* Librosa
* Soundfile

## Example eLEAF output

![Example eLEAF Output](https://github.com/amrosado/TF-EfficientLEAF/blob/aaron/initial_commit/example_output/eLEAF/eleaf_example_10.png?raw=true)

## Example Automatic Speech Recognition Transformer Output
```
target:     <forgive me i hardly know what i am saying a thousand times forgive me madame was right quite right this brutal exile has completely turned my brain>
prediction: <for get me i hardly no on what i am saying the thousand time me madame was right quite right as brittle excile as complictly turned my brain>

target:     <there cannot be a doubt he received you kindly for in fact you returned without his permission>
prediction: <their cannot be a doubt he received you kindly for in fact you returned without his permission>

target:     <oh mademoiselle why have i not a devoted sister or a true friend such as yourself>
prediction: <oh met must elie have i not a divoted sister or a true or a true friend the such as yourself>

target:     <what already here they said to her>
prediction: <what already here may said to her>

target:     <i have been here this quarter of an hour replied la valliere>
prediction: <i have been here this corder of an hour replied loviet lovieta loviety>

target:     <did not the dancing amuse you no>
prediction: <did not the dancing amuse you no>

target:     <no more than the dancing>
prediction: <no more then the dancing>

target:     <la valliere is quite a poetess said tonnay charente>
prediction: <the value is quite a poetice a tumish shonal up>

target:     <i am a woman and there are few like me whoever loves me flatters me whoever flatters me pleases me and whoever pleases well said montalais you do not finish>
prediction: <i am a woman and are few like me who ever loves me flatters me who have his me pleases me and who ever pleases well said not to lay you do not lay you do not finish>

target:     <it is too difficult replied mademoiselle de tonnay charente laughing loudly>
prediction: <it is ta difficult replied mudde muddenish all detenish all hall thing ladly>

target:     <look yonder do you not see the moon slowly rising silvering the topmost branches of the chestnuts and the oaks>
prediction: <look younder do you not saw rising slowly rising so hearing the top most branches of the chest and the oaks>

target:     <exquisite soft turf of the woods the happiness which your friendship confers upon me>
prediction: <exquisit soft turfable the woods the happiness which our friendship compars upon me>

target:     <well said mademoiselle de tonnay charente i also think a good deal but i take care>
prediction: <well said my must my should i all so sand i also think a good dial but i take care>

target:     <to say nothing said montalais so that when mademoiselle de tonnay charente thinks athenais is the only one who knows it>
prediction: <to say nothing said montally so there when mud mud was all deternatior obtanks at the nay is the only one who nose it>

target:     <quick quick then among the high reed grass said montalais stoop athenais you are so tall>
prediction: <quick then among the high read grassed montal a stoop at the nay you are so tall>

target:     <the young girls had indeed made themselves small indeed invisible>
prediction: <the hungerls had indeed made themselve small indeed and visible>

target:     <she was here just now said the count>
prediction: <she was her just now said the count>

target:     <you are positive then>
prediction: <you are positive then>

target:     <yes but perhaps i frightened her in what way>
prediction: <yes the perhaps are frightened what way>

target:     <how is it la valliere said mademoiselle de tonnay charente that the vicomte de bragelonne spoke of you as louise>
prediction: <how is it lowelly a said muddemose of the that the recompt a becompt a bright alone spoke a you as leas>

target:     <it seems the king will not consent to it>
prediction: <it seems the kingle not consemed to it>

target:     <good gracious has the king any right to interfere in matters of that kind>
prediction: <good gracians as the can any right to inner matters of that time>

target:     <i give my consent>
prediction: <i give my conset>

target:     <oh i am speaking seriously replied montalais and my opinion in this case is quite as good as the king's i suppose is it not louise>
prediction: <all i am speaking seriously replied montally and my opinion and this case his quite of the kings i suppose is a not louise is a not louise>

target:     <let us run then said all three and gracefully lifting up the long skirts of their silk dresses they lightly ran across the open space between the lake and the thickest covert of the park>
prediction: <what as run then sat all three and gracefully lifting up the long spirts of their so gresses they lightly ran crosseile conspace between the lake and the lake and the park>

target:     <in fact the sound of madame's and the queen's carriages could be heard in the distance upon the hard dry ground of the roads followed by the mounted cavaliers>
prediction: <and fact the sound of the dams and a queens carriages could be heard in the distance upon the hard dright round of the roads followed by the mountain cavaliers>

target:     <in this way the fete of the whole court was a fete also for the mysterious inhabitants of the forest for certainly the deer in the brake the pheasant on the branch the fox in its hole were all listening>
prediction: <in this way the fed of the whole core was a fet also for the misterious and habitants of the forest persertainly the dear in the break the fas intone the fox in its hole were all this and it holl listening>

target:     <at the conclusion of the banquet which was served at five o'clock the king entered his cabinet where his tailors were awaiting him for the purpose of trying on the celebrated costume representing spring which was the result of so much imagination and had cost so many efforts of thought to the designers and ornament workers of the court>
prediction: <at the conclusion of the bankly which was served it flive a court the cart the came in or his cavin or his tailors or awaiting him for the purpose of trying on the selebrated castoom representing spring which was the result of somewhat imagination and had cause some many of her suffers of thought to thes ts ous fin ofin th th the t cathathay omateschereley andgrd ocheraffocttoromugrmasle imaratinopatobinowledgatovide>

target:     <ah very well>
prediction: <how very will>

target:     <let him come in then said the king and as if colbert had been listening at the door for the purpose of keeping himself au courant with the conversation he entered as soon as the king had pronounced his name to the two courtiers>
prediction: <lien come and them said the king and as if cold bare had the most any the door for the purpose of keeping himself alcorance with the compersation he entered a soon as the king had pronounced his main to the shout corty years>

target:     <gentlemen to your posts whereupon saint aignan and villeroy took their leave>
prediction: <judgment dear posts were upon sand and and beautor thirly to thirly>

target:     <certainly sire but i must have money to do that what>
prediction: <certainly sire but i must have money to do that>

target:     <what do you mean inquired louis>
prediction: <what do mean in quartleries>

target:     <he has given them with too much grace not to have others still to give if they are required which is the case at the present moment>
prediction: <he has given them with two much grace not to have utterstolt to give if they are required which as the case of the present moment>

target:     <it is necessary therefore that he should comply the king frowned>
prediction: <it is messary their for that he should comply the king frowned>

target:     <does your majesty then no longer believe the disloyal attempt>
prediction: <those your magisty then no longer believe the disoilettent>

target:     <not at all you are on the contrary most agreeable to me>
prediction: <not it all you are on the conferry most agreeable to me>

target:     <your majesty's plan then in this affair is>
prediction: <your majesty plan then and this affair is>

target:     <you will take them from my private treasure>
prediction: <you will take them from my private treasure>

target:     <the news circulated with the rapidity of lightning during its progress it kindled every variety of coquetry desire and wild ambition>
prediction: <the new his circulated with the repetty elightening during of progress it candled every varied of coachery desire and while and while and bision>
```
