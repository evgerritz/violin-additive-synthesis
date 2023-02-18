(
var g3 = 196.0;
var c4 = 261.6;
var d4 = 293.665;
var e4 = 329.63;
var f4 = 349.228;
var g4 = 391.995;
var odeToJoy = [
	[e4, 1], [e4, 1], [f4, 1], [g4, 1], // 1
	[g4, 1], [f4, 1], [e4, 1], [d4, 1], // 2
	[c4, 1], [c4, 1], [d4, 1], [e4, 1], // 3
	[e4, 1.5], [d4, 0.5], [d4, 2], // 4

	[e4, 1], [e4, 1], [f4, 1], [g4, 1], // 5
	[g4, 1], [f4, 1], [e4, 1], [d4, 1], // 6
	[c4, 1], [c4, 1], [d4, 1], [e4, 1], // 7
	[d4, 1.5], [c4, 0.5], [c4, 2], // 8

	[d4, 1], [d4, 1], [e4, 1], [c4, 1], // 9
	[d4, 1], [e4, 0.5], [f4, 0.5], [e4, 1], [c4, 1], // 10
	[d4, 1], [e4, 0.5], [f4, 0.5], [e4, 1], [d4, 1], // 11
	[c4, 1], [d4, 1], [g3, 2], // 12

	[e4, 1], [e4, 1], [f4, 1], [g4, 1], // 5
	[g4, 1], [f4, 1], [e4, 1], [d4, 1], // 6
	[c4, 1], [c4, 1], [d4, 1], [e4, 1], // 7
	[d4, 1.5], [c4, 0.5], [c4, 2], // 8
];

~noteGen = {|freq=196, duration=1, amp=1.0, vibrato=0.5, pan=0.0|
	var g3_p_amps = [-32.4.dbamp, -9.5.dbamp, -20.8.dbamp, -20.3.dbamp, -19.2.dbamp, -20.9.dbamp, -26.1.dbamp, -27.2.dbamp, -23.9.dbamp, -22.1.dbamp, -31.1.dbamp, -34.1.dbamp, -28.2.dbamp, -31.6.dbamp, -37.8.dbamp, -35.9.dbamp, -30.5.dbamp, -30.9.dbamp, -31.2.dbamp, -53.6.dbamp, -53.2.dbamp, -50.2.dbamp, -46.8.dbamp, -54.2.dbamp, -49.3.dbamp, -51.0.dbamp, -72.1.dbamp, -56.0.dbamp, -52.9.dbamp, -56.9.dbamp];
	var num_partials = g3_p_amps.size;

	var overall_env = Env.linen(0.1, duration-0.1-0.1, 0.1, amp, \welch).kr();

	var violin = Array.fill(num_partials, {
		|i| p = i+1; SinOsc.ar(p*freq + SinOsc.kr(10*vibrato, mul:0.02*freq),
			mul: g3_p_amps[i]) //+ PinkNoise.kr(2, 0.000000001)
	});
	Out.ar(0, overall_env * Pan2.ar(Mix.ar(violin), pos:pan))
};

~playSong = {|melody, bpm|
	var running_time = 0.0;
	melody.do({ arg item;
		var note = item[0];
		var duration = item[1] * 60.0 / bpm;
		SystemClock.sched(running_time, {
			~noteGen.play(args: [\freq, item, \duration, duration, \vibrato, 0.5, \pan, 0.0]);
		});
		running_time = running_time + duration;
	})
};
FreqScope.new;
//~playSong.(odeToJoy, 144);
~noteGen.play(args: [\freq: 100, \duration: 3]);
)

(
FreqScope.new;
play { SinOsc.ar(440) };

)