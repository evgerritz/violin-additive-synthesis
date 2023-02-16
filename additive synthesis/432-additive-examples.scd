/*
 _______  ______   ______   ___   _______  ___   __   __  _______
|   _   ||      | |      | |   | |       ||   | |  | |  ||       |
|  |_|  ||  _    ||  _    ||   | |_     _||   | |  |_|  ||    ___|
|       || | |   || | |   ||   |   |   |  |   | |       ||   |___
|       || |_|   || |_|   ||   |   |   |  |   | |       ||    ___|
|   _   ||       ||       ||   |   |   |  |   |  |     | |   |___
|__| |__||______| |______| |___|   |___|  |___|   |___|  |_______|
 _______  __   __  __    _  _______  __   __  _______  _______  ___   _______
|       ||  | |  ||  |  | ||       ||  | |  ||       ||       ||   | |       |
|  _____||  |_|  ||   |_| ||_     _||  |_|  ||    ___||  _____||   | |  _____|
| |_____ |       ||       |  |   |  |       ||   |___ | |_____ |   | | |_____
|_____  ||_     _||  _    |  |   |  |       ||    ___||_____  ||   | |_____  |
 _____| |  |   |  | | |   |  |   |  |   _   ||   |___  _____| ||   |  _____| |
|_______|  |___|  |_|  |__|  |___|  |__| |__||_______||_______||___| |_______|
 _______  __   __  _______  __   __  _______  ___      _______  _______
|       ||  |_|  ||   _   ||  |_|  ||       ||   |    |       ||       |
|    ___||       ||  |_|  ||       ||    _  ||   |    |    ___||  _____|
|   |___ |       ||       ||       ||   |_| ||   |    |   |___ | |_____
|    ___| |     | |       ||       ||    ___||   |___ |    ___||_____  |
|   |___ |   _   ||   _   || ||_|| ||   |    |       ||   |___  _____| |
|_______||__| |__||__| |__||_|   |_||___|    |_______||_______||_______|

v022421

*/


/* ___________________________________________________

Example set 1: waveform generation

"https://en.wikibooks.org/wiki/Sound_Synthesis_Theory/Oscillators_and_Wavetables#Sine_wave"

The above has formula for basic waveform generation (sine, saw, square)

A cycle of a sine wave is 2pi radians long
Peak amplitude +/-1

To determine number of steps:

steps = S
cycle = T // one second
SampleRate = Fs

	S = T*Fs

To determine step size in radians:

    stepsize = 2pi*f / Fs

where f is the frequency.
___________________________________________________ */

// -- 1 --
// Input: Peak amplitude (a), Frequency (f)
// Output: Amplitude value (y)
(
l = List[];
a = 1; // peak amplitude
f = 1; // frequency :: one cycle -- change for more -- default 1
p = 0;	// phase -- change to pi/2 for cosine
r = 512; // samplerate -- default 512
)

r.do { y = a * sin(p) ; l.add( y ) ; p = p + ((2pi * f) / r) } ;
l.as(Array).plot;


// the below 'while' alternative won't work with alterations of phase or cycles above.  It will always stop before 2pi

while { p < 2pi } { l.add( a * sin(p) ) ; p = p + ((2pi * f) / r) }

l.asArray.plot ;


// -- 2 --
// Similarly, Square wave

l = List[];
p = 0;
while { p < 2pi } { if (p < pi) { l.add(a) } { l.add(a.neg)} ; p = p + ((2pi * f) / r) } ;
l.asArray.plot("Square!", minval: -1.1, maxval:1.1) ;


// -- 3 --
// same code, but 440 cps at samplerate, load to a buffer
(
l = List[];
a = 1; // peak amplitude
f = 440; // frequency :: one cycle -- change for more
p = 0;	// phase -- change to pi/2 for cosine
r = 44100; // samplerate
r.do { y = a * sin(p) ; l.add( y ) ; p = p + ((2pi * f) / r) } ;
)

b = Buffer.loadCollection(s, l.asArray, 1);
b.plot;

play { PlayBuf.ar(1, b, loop: true) }; // What happens if we change the rate?

play { PlayBuf.ar(1, b, 0.5, loop: true) }; // rate = 0.5 = down an octave

play { PlayBuf.ar(1, b, LFNoise0.kr(8, 1.5, 2), loop: true) }; // Use step noise to vary in a range

// -- 4a --
// An "arbitrary" waveform

(
l = List[];
a = 1; // peak amplitude
f = 1; // frequency :: one cycle -- change for more -- default 1
p = 0;	// phase -- change to pi/2 for cosine
r = 512; // samplerate -- default 512
)

r.do { y = a * sin(p) ; l.add( y + rrand(0.001, 0.1) ) ; p = p + ((2pi * f) / r) } ;

l.asArray.plot;

b = Buffer.loadCollection(s, l.asArray, 1);
b.play; // sort of a kick-drum

// -- 4b --
// A noisy sine :(

(
l = List[];
a = 1; // peak amplitude
f = 440; // frequency :: one cycle -- change for more
p = 0;	// phase -- change to pi/2 for cosine
r = 44100; // samplerate
r.do { y = a * sin(p) ; l.add( y + rrand(0.001, 0.1)) ; p = p + ((2pi * f) / r) } ;
)

l.asArray.plot; // can't really see the noise from this distance...
l.maxItem;
b = Buffer.loadCollection(s, l.asArray, 1);
b.normalize;
{ PlayBuf.ar(1, b) }.plot(0.01);
{ PlayBuf.ar(1, b, loop: true) }.play; // sine + noise.
play { PlayBuf.ar(1, b, LFNoise1.kr(5, 2, 2), loop: true) }; // What happens if we change the rate?

/* Question: Why do we perceive the above as two signals, a sine wave and some noise? */


/* ___________________________________________________

Example set 2 (after Slide 7): Cycle Generating methods and Useful Classes

The above code is simply to show different ways to generate waveforms from first principles. But it's silly and we don't do it. Below, we immediately see why Wavetable exists -- so the above isn't necessary.

___________________________________________________ */

// Wavetable.sineFill(size, amplitudes, phases)
Wavetable.sineFill(512, [0.5]).plot;

// sawtooth wave using sineFill, but with array with 1/n amplitudes

~saw = Array.fill(100, {|in| in = in + 1; 1.0/in});
~saw.plot;
x = Wavetable.sineFill(1024, ~saw).plot;


// Noise:

x = Wavetable.sineFill(8192, Array.fill(440, {rrand(0.5, 0.75)}), Array.fill(440, {rrand(0.0, 2pi)}));
x.plot;
x = x.asSignal; // convert to class Signal using 'asSignal' so we can call .play on it
x.play;


// Used with Shaper: shapes a SinOsc using the buffer "table data"

~saw = Array.fill(512, {|in| in = in +1; 1.0/in});
//~saw.plot;
b = Buffer.loadCollection(s, Wavetable.sineFill(512, ~saw));
(
{
    Shaper.ar(
        b,
        SinOsc.ar(300, 0, 1),
        0.5
    )
}.plot(0.00625); // change to .play to hear it.
)


/* Observations about using the above classes? */

/* Takeaways:

Lots of methods exist to transmute one collection type to another. Use these as starting points on a path of understanding what to use the different classes for. We will work with SineOsc below, but moving forward, you don't have to start with *nothing*. Sometimes it's better to start with something that's *almost* what you want and go from there.

*/


/* ___________________________________________________

Example set 3 (after Slide 9): Summing waveforms
___________________________________________________ */

// plot an array with first and second partials and their summed waveform.
(
k = []; // store the waveforms
a = 1; // peak amplitude
f = 2; // frequency
r = 512; // samplerate

// do structure to add first and second partial to an array
2.do {|i| i = i + 1;
	p = 0; // phase
	l = List[];
	r.do { y = a * sin(p) ; l.add( y ) ; p = p + ((2pi * (f * i)) / r) } ;
	k = k.add(l.asArray);
};
k = k.add(k[0] + k[1]).plot; //
)

// the same thing, but using Ugens
{ SinOsc.ar(400) }.plot

{ SinOsc.ar(400) + SinOsc.ar(800) }.plot

{ SinOsc.ar(400) + SinOsc.ar(800) + SinOsc.ar(1200) + SinOsc.ar(1600) }.plot // note peak amplitude...

// adjust the amplitude and play/plot it

{ SinOsc.ar(400, mul: 1) + SinOsc.ar(800,mul: 0.5) + SinOsc.ar(1200,mul: 0.25) + SinOsc.ar(1600,mul:0.125) }.plot

// make a sawtooth wave using only summed sines
(
{
	f = 200; // frequeny
	n = 16; // number of partials
	Mix.arFill(n, {|i| i = i + 1; SinOsc.ar(f * i, mul: 1 / i) });
}.plot
)

/* ___________________________________________________

Example set 4 (after Slide 10): Mix and Multichannel expansion
___________________________________________________ */

// sound the same but aren't
play { SinOsc.ar([400, 500], mul:0.5) } ;

play { Mix.ar( SinOsc.ar([400, 500], mul:0.5) ) } ;  // mix to single channel

// don't sound the same
play { SinOsc.ar([400, 500, 600], mul:0.33) }; // install rd_dot and try drawandplay
play { Mix.ar( SinOsc.ar([400, 500, 600], mul:0.33) ) } ;  // mix to single channel



/* ___________________________________________________

Example set 5 (after Slide 12): Envelopes

Envelopes are the primary means of controlling sound parameters in time for
dynamically changing sounds.

See here for a full explication and some additional classes useful
for GUI editing: https://www.scottericpetersen.com/2021/01/13/sculpting-gestural-space-with-envelopes-part-1/

___________________________________________________ */

//Env.linen(attackTime: 0.01, sustainTime: 1.0, releaseTime: 1.0, level: 1.0, curve: 'lin')

( // possibly useful env method for, oh, i don't know, homework...
{[
	Env.linen(0.1, 0.2, 0.3, 0.6).kr(),
	Env.linen(0.1, 0.2, 0.3, 0.6,\sine).kr(),
	Env.linen(0.1, 0.2, 0.3, 0.6,\welch).kr()]}.plot(0.6);
)

// get interpolated values
e = Env.linen(0.1, 0.8, 0.1, 0.8);
e[0.98999]; // type some float here between 0.0 and 1.0

// amp and freq control
e = Env.linen(1, 2, 1, 0.6);
f = Env.new([rrand(50,440), rrand(1000, 1500), rrand(1000, 2000), rrand(40,440)],[1, 2, 1]);
// .kr to play on server
play { SinOsc.ar(f.kr, mul: e.kr) } ;

// or...

~fund = 400;
a = Array.fill(50, {|in| in = in + 1; ~fund * in }); // freq array
// *or*
a = Array.fill(50, {|in| in = in + 1; if (in.odd, {~fund * in }, {0}) }); // odd-harmonic freq array

b = Array.fill(50, {|in| in = in +1; if (in.odd, {1/in }, {0}) }); // amplitude array
// *or*
b = Array.fill(50, {|in| in = in +1; 1/in }); // amplitude array

f = Env.new([1, 1.4, 1.4, 1],[1, 2, 1]); // frequency envelope
e = Env.linen(1, 2, 1, 0.6); // our old friend e

scope { Mix.ar( SinOsc.ar(a * f.kr, mul: b * e.kr) ) } ;


/* ___________________________________________________

Example set 6 (after Slide 14): LFOs
___________________________________________________ */

// plot some available primary waveforms

{[SinOsc.ar, LFSaw.ar, LFTri.ar, LFPulse.ar, LFNoise1.ar(3000), LFNoise0.ar(3000)]}.plot;

// Ubiquitous vibrato example

play { SinOsc.ar(200 + SinOsc.ar(5, mul:20) )} ;

// 2x per second, the signal will "travel" amplitude val around 0

// note static value for mul is bad. Q. Why? (Hint: change frequency value in SinOsc and listen)


// mul as percentage of frequency fixes problem

x = {|freq=440| SinOsc.ar(freq + SinOsc.ar(5, mul: 0.01 * freq) )}.play ;  // freq is wut?

x.set(\freq, rrand(400, 2400)) ;


// adjust LFNoise1.kr(2, THISVAL * freq, 0) to be larger or smaller to change how well the "performer" can hold her pitch.

x = {|freq=440| SinOsc.ar((freq + LFNoise1.kr(2, 0.005 * freq, 0)) + SinOsc.ar(5, mul: 0.01 * freq) )}.play ;  // freq is wut?

x.set(\freq, rrand(400, 2400)) ;



// Low-frequency Noise used to add some complexity...

~fund = 400;
a = Array.fill(50, {|in| in = in + 1; ~fund * in }); // freq array: TABLE
b = Array.fill(50, {|in| in = in +1; 1/in }); // amplitude array: TABLE
f = Env.new([1, 1.4, 1.4, 1],[1, 2, 1]); // frequency array
e = Env.linen(1, 2, 1, 0.6); // amplitude envelope

play { Mix.ar( SinOsc.ar((a * LFNoise1.kr(1, 0.009, 1) ) * f.kr, mul: (b * e.kr) * LFNoise1.kr(1, 0.01, 1)) ) } ;


// the above still applies a single random variation to all frequencies - below applies them individually

// something from history
(
SynthDef("droney", {arg freq=50, mx=0.3, lvl=0.5, tms=5;
	var env, lfof, lfoa, saws, sound, vrb;

	env = EnvGen.kr(Env.new([0, lvl, lvl, 0], [tms, tms-2, tms]), doneAction:2);
	lfof = Array.fill(5, {LFNoise1.kr(1, 0.005*freq, 0) + PinkNoise.kr(0.008)});
	lfoa = Array.fill(5, {LFNoise1.kr(15, 0.1, 1)});
	saws = Array.fill(5, {|i| Saw.ar(freq + lfof[i], env*lfoa[i]) + RLPF.ar(PinkNoise.ar(),freq,0.01, 0.01)});
	sound = Mix(saws);
	vrb = FreeVerb.ar(Splay.ar(RLPF.ar(saws, 1000)), mx);
	//lfof.poll;
	Out.ar(0, vrb);

}).add;
)

// whack this a few times
Synth(\droney, [\freq, rrand(200, 1200).round(100)]);

// some lowness

Synth(\droney, [\freq, rrand(50, 200).round(50)]);

// some lowerness

Synth(\droney, [\freq, 5]);



/* Think Pair Share: Thought Experiment (that might turn into a Pset):

How many envelopes would be required to synthesize a string-like instrument

You must account for:

Transient (affects amplitude and frequency)
Frequency and Amplitude variation for all partials
Vibrato
Overall Amplitude (that affects high frequencies more than low frequencies...)

*/





/* ___________________________________________________

Example set 7 (after Risset, Mutations): "Bells gone Wild"
___________________________________________________ */

(
play {
s
	var fund, nPar, aa, fa, ta, pa, ar;
	fund = 100;
	nPar = 10;
	aa = List[ 0.68, 0.21, 0.25, 0.05 ] ; // amplitude array
	fa = List[ 1.1, 1.08, 1.0, 1.0 ]; // frequency array: fund multiplier
	ta = [ 0.01, 0.1, 1.8 ];  // time array
	pa = Array.fill(nPar, {|i| i = i + 1; fund * (i + rrand(-0.1, 0.279)) });// partial array


		ar = Array.fill(nPar, {|i| var laa, lfa, lta;

		laa = aa.copy; // w/out copy, laa will literally be the array stored in aa
		lfa = fa.copy;
		lta = ta.copy;

		laa.isArray.postln;

		(nPar - aa.size).do {	laa.add(rrand(0.05, 0.65)); lfa.add(rrand(0.08, 0.65)); lta.add(rrand(0.05,0.57)) };
		laa.put((laa.size - 1), 0.0001) ; // replace the last value with (almost) silence

		lfa.postln;
		//pa[i].postln;

		Pan2.ar(SinOsc.ar(pa[i] * EnvGen.kr(Env(lfa, lta)), 0, EnvGen.kr(Env(laa, lta))), rrand(-0.5, 0.5)) ;

	} ) ;

	ar.postln;
	Mix(ar) * 0.5; // mix down to single channel
}
)


s.makeGui




// _______________________________________________

// More examples, from Cottle: CM with Logic and SC3

(
// Saw spread across channels. Use 's' to overlay.
a = {Array.fill(16, {arg ct;
SinOsc.ar(400 * (ct+1), phase: 0,
mul: (1/(ct+1)))}) * 0.6 };
a.plot; a.play;
)

(
// With random phase
a = {Mix.fill(16, {arg ct;
SinOsc.ar(400 * (ct+1), phase: pi.rand,
mul: (1/(ct+1)))}) * 0.6 };
a.plot; a.play )


// Cottle 22.3 Additive Saw Modulation

(
{
var rate = 14;
f = 150;
t = Impulse.kr(1/3); Mix.fill(12, {arg ct;
Pan2.ar(SinOsc.ar(ct+1*f,
mul: max(0, LFNoise1.kr(rrand(rate, rate*2))/1)), 1.0.rand2)
})*0.1 }.scope(1)
)

(
{ //let it run a while
var rate = 6, globalBend;
f = 150;
globalBend = max(0, LFNoise1.kr(1/5, 0.7, -0)); t = Impulse.kr(1/3);
Mix.fill(12, {arg ct;
var bend;
bend = 1 + (globalBend * rrand(-1.0, 1.0)); Pan2.ar(SinOsc.ar(ct+1*(f*bend),
mul: max(LFTri.kr(1/10, mul: 0.3, add: 0.3), LFNoise1.kr(rrand(rate, rate*2))/1)), 0.2.rand2)
})*0.1 }.scope(1)
)

