#!/usr/bin/perl
use strict;
use warnings;

for (my $hidden=4; $hidden<=20; $hidden+=4) {
	for (my $lambda=0.0; $lambda<=1.0; $lambda+=0.2) {
		system 'python nnScript.py '.$hidden.' '.$lambda.' >> results.log 2>&1';
	}
}
