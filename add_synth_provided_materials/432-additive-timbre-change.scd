/* 432/532: In-class exposition of loudness/brightness relationship with two examples */


/* METHOD 1: USING UGENS/METHODS TO CREATE CURVE INSIDE SYNTH */

// Example showing use of LinExp and LinLin Ugens
(
play {
	var ctl_env = Env([0.0, 1, 0.0], [2, 2]).kr(2);

	var exp = SinOsc.ar(LinExp.kr(ctl_env,0.0, 1.0, 400, 800));
	var lin = SinOsc.ar(LinLin.kr(ctl_env,0.0, 1.0, 400, 800));
	Out.ar(0, [lin, exp]);
}
)

// Example showing use of lincurve method (SimpleNumber, SequenceableCollection)

(
 {
	var ctl_env = Env([0.0, 1, 0.0], [2, 2]).kr(2);

	var lin = SinOsc.ar(ctl_env.lincurve(0.0, 1.0, 400, 800, -4));
	var exp = SinOsc.ar(ctl_env.lincurve(0.0, 1.0, 400, 800, 4));
	Out.ar(0, [lin, exp]);
}.play
)

(
 {
	var ctl_env = Env([0.0, 1, 0.0], [2, 2]).kr(2);

	var lin = SinOsc.ar(ctl_env.lincurve(0.0, 1.0, 400, 800, -4));
	var exp = SinOsc.ar(ctl_env.lincurve(0.0, 1.0, 400, 800, 4));
	[ctl_env.lincurve(0.0, 1.0, 400, 800, -4), ctl_env.lincurve(0.0, 1.0, 400, 800, 4)];
}.plot(4)
)


{ Env([0,1,0],[0.15, 0.15], [4, -4]).kr(2) }.plot(0.3)



/* METHOD 2: SUPPLYING SCALED VALUES TO SYNTH AT CREATION TIME */


/* Create two amplitude spectrum arrays */
~num_harms = 40;

(
~random_amps_gen = {Array.fill(~num_harms, {|i| var amp, amp_mod;
	i = i + 1;
	amp = 1/i; // one over increment = sawtooth if no changes made
	amp_mod = 0.8 * amp;
	amp + rrand(amp_mod.neg, amp_mod);
})};

~array1 = ~random_amps_gen.();

~array2 = ~random_amps_gen.();

[~array1, ~array2].plot;

)

/* Create an Env for each partial that starts with the value of array 1 and ends with value of array 2 */

~amp_envs = Array.fill(~array1.size, {|i| Env([~array1[i], ~array2[i]], 7) });

~amp_envs.plot;

(
fork {

	var scale = ([ 0, 2, 4, 5, 7, 9, 11, 12 ] + 55).midicps; // major scale starting on C4/Middle C

	8.do {|in|
		play {
			Mix.arFill(~num_harms, {|i| i = i + 1;
				i.postln;
				SinOsc.ar(scale[in] * i, 0, ~amp_envs[i - 1][in]) } * Env.perc.kr(2);
			)
		};
		1.wait;
	}
}

)