/* Effects of LFOs on a Signal */

/* -- PART 1 -- LFO Summed with (or substituted for) frequency

The output of any low frequency object, be it oscillator or noise, is a stream of floats representing, usually, amplitudes.

The amplitude (mul argument) of the LF object determines the peak deviation from the 'add' component. In typical use, say with SinOsc.ar, we do not do anything with 'add' because adding or subtracting from it changes the signal from bipolar towards unipolar.

-- 1a --

SinOsc.ar(1, 0, 1, 0); results in a peak deviation of 1 around 0, resulting in -1 to 1 output.
*/

{ SinOsc.ar(400, 0, 1, 0) }.plot(1/400);


/* -- 1b --

Similarly, if the amplitude is 2, the output will be a peak deviation of 2, resulting in -2 to 2 output.

*/

{ SinOsc.ar(400, 0, 2, 0) }.plot(1/400);

/* -- 1c --  Adding offset using the 'add' argument

If we add a constant value to 'add', the rest state or center of the deviation will be shifted to that point.

Ex. set mul to 1 and add to 1 for a peak deviation of 1 around 1. The output will be 0-2. It is not a unipolar signal.

*/

{ SinOsc.ar(400, 0, 1, 1) }.plot(1/400);

/* -- 1c --  Adding offset with LF noise Ugens

The same is true even if the signal is not periodic. Using LFNoise0, which is sample and hold, or step noise, we will set mul to 1 and add to 1 as above. While the results are pseudorandom, the peak deviation of 1 around an offset or center of 1 results in a *possible* output range of 0 to 2.

Here, we will capture 5 "cycles" of the LFNoise0 object. Of course, the plot will be different every time we evaluate the code.

*/

{ LFNoise0.ar(400, 1, 1) }.plot(1/400 * 5);


/* -- 1c --  Summing an LFO with a SinOsc frequency argument.

This process results in a "vibrato" effect, a deviation of the pitch around a center frequency. The center is described by the frequency argument (a float or int). The deviation distance around that point is described by the amplitude of the LFO as shown above. The number of deviations per second is determined by the frequency argument of the LFO.

Note: on a time-plot of frequency change, you must examine the width of the waveform for frequency change. It is much harder to see than amplitude change, but easy to hear once the deviation approaches a semi-tone or fifty cents from the center frequency.

*/

{ SinOsc.ar(400 + SinOsc.ar(5, 0, 10, 0)) }.play;


/* -- 1c --  Summing an LF noise object with a SinOsc frequency argument.

The output here is as above in terms of peak deviation (10) and number of times per-second the change should occur (5.)

*/

{ SinOsc.ar(400 + LFNoise1.ar(5, 10, 0)) }.play;


/* -- PART 2 -- Multiplying a signal with an LFO

Here, the deviation and center (add) rules from above still apply, but instead of summing this output with a constant as frequency input argument to a signal, we will multiply the signal and the LFO.

*/

/* -- 2a -- Multiplying Signals: the amplitude rule

Unlike above, once we are multiplying amplitudes we must adhere to rule 1 of sound synthesis -- never exceed maximum amplitude or 1. If we were to use .play on the below example our signal would be 10 times the allowable output level and would potentially damage our ears and/or our audio equipment.

Here we plot the output showing the input signal with an amplitude of 1 being multiplied by a signal with an amplitude of 10. The output is, therefore, a peak amplitude of 10.

WARNING: DO NOT USE .play ON THE BELOW EXAMPLE. USE .plot ONLY
*/

{ SinOsc.ar(400) * SinOsc.ar(5, 0, 10, 0) }.plot(1/2);



/* -- 2b -- Multiplying Signals: proper scaling

The rule for multiplying signals is actually easier than summing signals: any number of signals can be multiplied together as long as their individual amplitudes are no higher than 1.

Here, the rapid (5 times per second) oscillation of the LFO will result in a quick, obvious tremolo effect.
*/

{ SinOsc.ar(400) * SinOsc.ar(5, 0, 1, 0) }.plot(1/2); // plot

{ SinOsc.ar(400) * SinOsc.ar(5, 0, 1, 0) }.play; // play


/* -- 2c -- Multiplying Signals: LF noise with low amplitude for "humanization"

One obvious observation of computer-produced sound is that it is regular in a way that humans simply cannot produce on an instrument or with their voices. The perfection of the output can be desirable or undesirable depending on your application. If mimicking real instruments is a goal, or if the "perfect" nature of the algorithm sounds sterile, boring, or "fake", this can be remedied with a small amount of noise.

Here, we use LFNoise1 (ramp noise) to inject a slight amount of randomness to the amplitude of a signal. we must add 1 minus the peak deviation to the offset (add) of the LF object if we want the signal to vary close to 1.

Adjust the mul argument to increase or reduce the peak deviation. Too much and the sound will be "wobbly". Too little and you will not notice it.
*/

{ SinOsc.ar(400) * LFNoise1.ar(5, 0.025, 1 - 0.025) }.plot(1/20); // plot

{ SinOsc.ar(400) * LFNoise1.ar(5, 0.025, 1 - 0.025) }.play; // plot


/* -- 3 -- Putting it all together.

Neither of the LFO strategies above account for the overall shape of the sound. For that, we need envelopes, introduced in the 432-additive-examples.scd document. Below we show an example of using LF Noise and LFOs to:

- Create a vibrato
- Randomize the frequency slightly: this is a product of a small number * the input frequency
- Randomize the amplitude slightly

And we control the overall shape using a standard envelop shape.

*/
(
play {|pitch = 400, vib_spd = 5|

	var f_rand = LFNoise1;
	var f_vib = SinOsc;
	var a_rand = LFNoise1.kr(5, 0.025, 0.975);
	var snd = SinOsc.ar(pitch + f_vib.kr(vib_spd, 0, 0.02 * pitch) + f_rand.kr(5, 0.01 * pitch));
	var env = Env.linen().kr(2);
	snd * env;
}
)