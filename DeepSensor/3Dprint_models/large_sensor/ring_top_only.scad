// OpenSCAD input script
// Created using Claude AI with lot of instructions and fixes.

// SCAD parameter "face number" to approximate ciylinders
$fn = 100;


// Parameters
main_height = 100;
outer_diameter = 118;
inner_diameter = 32.5;
wall_thickness = 5;
num_ribs = 3;
cut_diameter = outer_diameter - 30;
rib_height = 30;

// Spring parameters
spring_thickness = 2;
spring_height = 30;
//spring_width = 30;
spring_gap = 2;
spring_extension = 4;
n_slices = 10;

// strip holes parameters
strip_width = 2;
strip_height = main_height - 2 * rib_height;


strip_angles_ = [0, 180, 300, 270, 350];
angle_zero = 20;
strip_angles = concat(
    [for (a = strip_angles_) angle_zero - a],
    [for (a = strip_angles_) angle_zero + a]);

// Derived parameters
outer_radius = outer_diameter / 2;
inner_radius = inner_diameter / 2;
middle_radius = (outer_radius + inner_radius) / 2;
cut_radius = cut_diameter / 2;

// Calculated spring position
//function chord_height(r, w) = r - sqrt(r*r - (w/2)*(w/2));
//spring_chord_height = chord_height(outer_radius, spring_width);
//spring_offset = outer_radius - spring_thickness/2 - spring_chord_height;


module cylinder_shell(height, radius, thickness) {
    difference() {
        cylinder(h = height, r = radius, center = true);
        cylinder(h = height + 1, r = radius - thickness, center = true);
    }
}

// Outer cylinder shell
module outer_shell() {
    difference() {
        cylinder_shell(main_height, outer_radius, wall_thickness);
        for(angle = strip_angles) {
            rot_angle = (180 / num_ribs + angle) % 360;
            rotate([0, 0, rot_angle])
            translate([outer_radius, 0, 0])
            cube([3*wall_thickness, strip_width, strip_height], center = true);
        }    
    }
}

// Inner cylinder shell
module inner_shell() {
    translate([0, 0, (spring_height +  spring_gap)/2])
    cylinder_shell(main_height + spring_height +  spring_gap, inner_radius + wall_thickness, wall_thickness);
}

// Single rib
module rib() {
    translate([inner_radius, -wall_thickness/2, -main_height/2])
    cube([outer_radius - inner_radius - wall_thickness/2, 
        wall_thickness, 
        main_height]);
}

// Cut cylinder
module cut_cylinder() {
    cylinder(h = main_height - 2*rib_height, r = cut_radius, center = true);
}

// Single spring - box approximation
module spring_approx() {
    cube([spring_thickness, spring_width, spring_height], center = true);
}

/*
module spring_vertical_cut() {
    L = spring_height;
    D = spring_extension;
    radius = (L*L + 4 * D*D) / (8 * D); 
    angle = 2 * acos( (radius - D) / radius );
    
    translate([-(radius - D), 0])
    intersection() {
        difference() {
            circle(radius);
            circle(radius - spring_thickness);
        }
        translate([radius/2, 0])
        square([radius, spring_height], center=true);
    }
}

// Simple planar spring (vertically bent)
module spring() {
    spring_angle = spring_width / outer_radius / 2 / PI * 180;
    rotate_extrude(angle=spring_angle, $fn=100) {
        translate([outer_radius, 0])
        //rotate([90, 0, 0])
        spring_vertical_cut();
    }
}

// Simple planar spring (vertically bent)
module spring_v() {
    points = [
        [-spring_thickness/2, -spring_height/2],
        [spring_thickness/2, -spring_height/2],
        [spring_thickness/2 + spring_extension, 0],  // Bend point
        [spring_thickness/2, spring_height/2],
        [-spring_thickness/2, spring_height/2],
        [-spring_thickness/2 + spring_extension, 0]  // Bend point on the other side
    ];
    
    spring_angle = spring_width / outer_radius / 2 / PI * 180;
    rotate_extrude(angle=spring_angle) {
        translate([outer_radius - spring_thickness/2, 0])
        //rotate([90, 0, 0])
        polygon(points);
    }
}
*/


/*
// Calculate the radius of the spring cylinders
spring_radius = (outer_radius + spring_extension) / (1 + 2 / sqrt(3));

// Function to create a single horizontal cut with varying r_x
module barrel_spring_cut(r_x) {
    union() {
        // Semi-circle result
        difference() {
            circle(spring_radius, $fn=20);
            translate([spring_radius / 2, 0])
            square([spring_radius, 2 * spring_radius], center = true);
        }

        // Ellipse for X>0
        scale([r_x, 1])
        circle(spring_radius, $fn=20);
    }
}

// Function to calculate r_x based on z
function r_x(z) = 1 - (spring_radius - spring_extension) / spring_radius * pow(z / spring_height, 2);

// Module to create the deformed barrel spring using hull operation
module spring_barrel() {
    hull() {
        for (z = [-spring_height/2 : spring_height/n_slices : spring_height/2]) {
            translate([0, 0, z])
            linear_extrude(height = spring_thickness)
            barrel_spring_cut(r_x(z));
        }
    }
}


module horizontal_spring() {
    inner_scale = (spring_radius - spring_thickness) / spring_radius;
    difference(){
        spring_barrel();
        scale([inner_scale, inner_scale, 1])
        spring_barrel();
    }
} */

function r_for_chord(chord, extension) = (chord*chord/4 + extension*extension) / 2 / extension;

module horizontal_spring() {
    // three touching springs, without extension
    spring_radius = outer_radius / (1 + 2 / sqrt(3));
    contact_bump_r = r_for_chord(spring_height, spring_extension)*2/3;

    spring_z_shift = main_height/2 + spring_height/2 + spring_gap;
    translate([2/sqrt(3) * spring_radius, 0, 
                spring_z_shift])
    difference() {
        union() {
            cylinder(h=spring_height, r=spring_radius, center=true);
            translate([spring_radius - contact_bump_r + spring_extension, 0, 0])
            sphere(r=contact_bump_r);
        }
        cylinder(h=4*spring_height, r=spring_radius - spring_thickness, center=true);
    }
    
}

// Main assembly
module main_assembly() {
    union() {
        difference() {
            union() {
                outer_shell();
                for (i = [0:num_ribs-1]) {
                    angle = i * 360 / num_ribs;
                    rotate([0, 0, angle])
                    rib();
                }
                // Add springs
                for (i = [0:num_ribs-1]) {
                    angle = i * 360 / num_ribs;  // Offset by half the angle between ribs
                    rotate([0, 0, angle  + 180 / num_ribs])
                    horizontal_spring();
                    rotate([0, 0, angle])
                    translate([0, 0, spring_height+spring_gap])
                    difference(){
                        rib();
                        translate([outer_radius*0.4, -outer_radius/2, 0])
                        cube([outer_radius, outer_radius, outer_radius]);
                    }
                }
            }
            union() {
                cut_cylinder();
                cylinder(h=3*main_height, r=inner_radius + wall_thickness/2, center=true);
            }
        }
        inner_shell();
    }   
}

 
// Render the assembly
difference() {
    main_assembly();  
    translate([0, 0, -main_height/2])
    cube([3*outer_radius, 3*outer_radius, main_height], center=true);
    translate([0, 0, +main_height])
    cube([3*outer_radius, 3*outer_radius, main_height], center=true);
}
