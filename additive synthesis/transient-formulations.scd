/* Transient Formulations

Below are four examples of using noise and periodic signals as transient multipliers for N synthesis type.

*/


( // 1. Noise transient as amplitude modulator (note DC offset of 1)
play {
	var transient = (PinkNoise.ar( Env.linen(0.001, 0.01, 0.001).kr, 1 ));
	var envelope = Env.perc.kr;
	SinOsc.ar(rrand(200, 600), 0, transient * envelope);
}
)


( // 1b. Noise transient as amplitude modulator (note DC offset of 1) & Frequency modulator
play {
	var transient = (PinkNoise.ar( Env.linen(0.001, 0.01, 0.001).kr, 1 ));
	var envelope = Env.perc.kr;
	SinOsc.ar(rrand(200, 600) * transient, 0, transient * envelope);
}
)

(
/* 2. Sinusoidal transient as amplitude modulator (note DC offset of 1)

1. Try experimenting with the frequency value of the transient SinOsc
2. Try experimenting with the amplitude value for the transient SinOsc

*/

play {
	var transient = (SinOsc.ar(10, 0, Env.linen(0.001, 0.01, 0.001).kr, 1 ));
	var envelope = Env.perc.kr;
	SinOsc.ar(rrand(200, 600), 0, transient * envelope);
}
)

(
/* 2b. Sinusoidal transient as amplitude modulator (note DC offset of 1)

1. Try experimenting with the frequency value of the transient SinOsc
2. Try experimenting with the amplitude value for the transient SinOsc

*/

play {
	var transient = (SinOsc.ar(10, 0, Env.linen(0.001, 0.01, 0.001).kr, 1 ));
	var envelope = Env.perc.kr;
	SinOsc.ar(rrand(200, 600) * transient, 0, transient * envelope);
}
)


// FM Trumpet Using Sinusoidal Transient (tra) with Line

(
SynthDef(\trumpet, { |cf = 400, mf = 1, im = 1000, fb = 100|

	var n, e, en, m, c, p, lo, li, lfp, lfa, tra;

	tra = SinOsc.kr(25, 0, Line.kr(0.5,0,0.125), 1); // transient
	lfp = LFNoise0.kr(10, 0.001 * cf);  // microrandomness for pitch
	lfa = LFNoise1.kr(10, 0.01, 1);  // microrandomness for amplitude
	e = Env.linen(releaseTime:0.1); // amplitude envelope
	li = LocalIn.ar(1) * fb;  // feedback path
	m = SinOsc.ar(cf * mf, 0, im * (e.kr(2) + 1));  // modulator signal
	c = SinOsc.ar(cf + lfp + m) * e.kr(2) * lfa * tra;  // carrier signal
	n = BPF.ar(WhiteNoise.ar(), cf * 2, 0.05, 0.6) * e.kr * lfa * tra; // noise simulating breath
	lo = LocalOut.ar(c); // output for feedback loop
	p = Pan2.ar(c + n); // panning (of carrier and noise)

	Out.ar(0, p);

}).add;
)

Synth(\trumpet, [\cf, rrand(54, 80).midicps, \mf, 1]);