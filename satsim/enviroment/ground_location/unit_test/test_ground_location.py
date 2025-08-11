import pytest
import torch
import numpy as np
import math
from satsim.enviroment.ground_location import (
    GroundLocation,
    GroundLocationStateDict,
    SpaceCraftStateDict,
    Ephemeris,
)
from satsim.architecture.timer import Timer


def test_range():
    """
    Tests whether groundLocation:
    1. Computes range correctly by evaluating slantRange
    2. Tests whether elevation is correctly evaluated
    3. Tests whether range limits impact access
    4. Tests whether multiple spacecraft are supported in parallel
    """

    # Create planet state
    planet_state = Ephemeris({
        'position_vector': torch.zeros(3),
        'velocity_vector': torch.zeros(3),
        'J2000_2_planet_fixed': torch.eye(3),
        'J2000_2_planet_fixed_dot': torch.zeros(3, 3),
    })

    # Initialize ground station
    ground_location = GroundLocation(
        planet_radius=6378.137e3,  # Earth radius [m]
        maximum_range=100e3,  # Maximum range [m]
        minimum_elevation=80.0 * torch.pi /
        180.0,  # 80 degrees minimum elevation
        timer=Timer(dt=1.0))

    # Set ground station location (equator, 0 degrees longitude)
    position_LP_P_init, direction_cosine_matrix_Planet2Location = ground_location.specify_location(
        latitude=torch.tensor(0.0),  # 0 degrees latitude [rad]
        longitude=torch.tensor(0.0),  # 0 degrees longitude [rad]
        altitude=torch.tensor(0.0)  # altitude [m]
    )

    # SC3: Within height range, but insufficient elevation (79 degrees elevation)
    # Use Basilisk's euler3 rotation matrix
    angle_rad = torch.tensor(11.0 * torch.pi / 180.0)  # 11 degrees to radians
    rotation_matrix = torch.tensor(
        [[torch.cos(angle_rad),
          torch.sin(angle_rad), 0.0],
         [-torch.sin(angle_rad),
          torch.cos(angle_rad), 0.0], [0.0, 0.0, 1.0]])
    # First rotate [100e3, 0, 0], then add Earth center position
    position_rotated = rotation_matrix @ torch.tensor([100e3, 0.0, 0.0])
    position_tmp = torch.tensor([6378.137e3, 0.0, 0.0]) + position_rotated

    # Create spacecraft states (batch format)
    spacecraft_states = SpaceCraftStateDict({
        'position_BN_N':
        torch.stack([
            torch.tensor([6378.137e3 + 100e3, 0.0,
                          0.0]),  # SC1: Within range, 100km height
            torch.tensor([6378.137e3 + 101e3, 0.0,
                          0.0]),  # SC2: Out of range, 101km height
            position_tmp,  # SC3: Within height range, but insufficient elevation (79 degrees elevation)
        ]),
        'velocity_BN_N':
        torch.zeros(3, 3),
        'position_CN_N':
        torch.stack([
            torch.tensor([6378.137e3 + 100e3, 0.0, 0.0]),
            torch.tensor([6378.137e3 + 101e3, 0.0, 0.0]),
            position_tmp,
        ]),
        'velocity_CN_N':
        torch.zeros(3, 3),
        'sigma_BN':
        torch.zeros(3, 3),
        'omega_BN_B':
        torch.zeros(3, 3),
        'omegaDot_BN_B':
        torch.zeros(3, 3),
    })

    # Initialize state dictionary
    state_dict = ground_location.reset()

    # Run calculation
    result = ground_location(state_dict, spacecraft_states, planet_state,
                             position_LP_P_init,
                             direction_cosine_matrix_Planet2Location)

    # Get results
    access_states = result['access_states']

    # Verify results
    accuracy = 1  # Allow 1m error

    # Expected values
    expected_ranges = torch.tensor([100e3, 101e3, 100e3])
    expected_elevations = torch.tensor(
        [torch.pi / 2, torch.pi / 2,
         79.0 * torch.pi / 180.0])  # 90 degrees, 90 degrees, 79 degrees
    expected_access = torch.tensor(
        [1, 0, 0], dtype=torch.uint8
    )  # Has access, no access, no access (79 degrees < 80 degrees limit)

    # Check results
    assert torch.allclose(access_states['slant_range'],
                          expected_ranges,
                          atol=accuracy), "Range mismatch"
    assert torch.allclose(access_states['elevation'],
                          expected_elevations,
                          atol=accuracy), "Elevation mismatch"
    assert torch.allclose(access_states['has_access'],
                          expected_access), "Access mismatch"

    print("✅ Range test passed!")


def test_rotation():
    """
    Tests whether groundLocation computes the current location based on 
    the initial position and the rotation rate of the planet
    """

    # Create planet state (rotated -10 degrees)
    angle = torch.tensor(-10.0 * torch.pi / 180.0)
    rotation_matrix = torch.tensor([[torch.cos(angle),
                                     torch.sin(angle), 0.0],
                                    [-torch.sin(angle),
                                     torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])

    planet_state = Ephemeris({
        'position_vector': torch.zeros(3),
        'velocity_vector': torch.zeros(3),
        'J2000_2_planet_fixed': rotation_matrix,
        'J2000_2_planet_fixed_dot': torch.zeros(3, 3),
    })

    # Initialize ground station
    ground_location = GroundLocation(planet_radius=6378.137e3,
                                     maximum_range=200e3,
                                     minimum_elevation=10.0 * torch.pi / 180.0,
                                     timer=Timer(dt=1.0))

    # Set ground station location (0 degrees latitude, 10 degrees longitude)
    position_LP_P_init, direction_cosine_matrix_Planet2Location = ground_location.specify_location(
        latitude=torch.tensor(0.0),
        longitude=torch.tensor(10.0 * torch.pi / 180.0),
        altitude=torch.tensor(0.0))

    # Create spacecraft state
    spacecraft_states = SpaceCraftStateDict({
        'position_BN_N':
        torch.tensor([6378.137e3 + 90e3, 0.0, 0.0]),
        'velocity_BN_N':
        torch.zeros(3),
        'position_CN_N':
        torch.tensor([6378.137e3 + 90e3, 0.0, 0.0]),
        'velocity_CN_N':
        torch.zeros(3),
        'sigma_BN':
        torch.zeros(3),
        'omega_BN_B':
        torch.zeros(3),
        'omegaDot_BN_B':
        torch.zeros(3),
    })

    # Initialize state dictionary
    state_dict = ground_location.reset()

    # Run calculation
    result = ground_location(state_dict, spacecraft_states, planet_state,
                             position_LP_P_init,
                             direction_cosine_matrix_Planet2Location)

    # Get results
    access_states = result['access_states']

    # Verify results
    accuracy = 1e-6

    expected_range = torch.tensor(90e3)
    expected_elevation = torch.tensor(torch.pi / 2)  # 90 degrees
    expected_access = torch.tensor(1, dtype=torch.uint8)

    assert torch.allclose(access_states['slant_range'],
                          expected_range,
                          atol=accuracy), "Range mismatch"
    assert torch.allclose(access_states['elevation'],
                          expected_elevation,
                          atol=accuracy), "Elevation mismatch"
    assert torch.allclose(access_states['has_access'],
                          expected_access), "Access mismatch"

    print("✅ Rotation test passed!")


def test_azimuth_elevation_range_rates():
    """
    Tests whether groundLocation computes azimuth, elevation, and range rates correctly
    """

    # Create planet state
    planet_state = Ephemeris({
        'position_vector': torch.zeros(3),
        'velocity_vector': torch.zeros(3),
        'J2000_2_planet_fixed': torch.eye(3),
        'J2000_2_planet_fixed_dot': torch.zeros(3, 3),
    })

    ground_location = GroundLocation(planet_radius=6378.137e3,
                                     minimum_elevation=60.0 * torch.pi / 180.0,
                                     timer=Timer(dt=1.0))

    # Set ground station location (0 degrees latitude, 0 degrees longitude) - Equator position
    position_LP_P_init, direction_cosine_matrix_Planet2Location = ground_location.specify_location(
        latitude=torch.tensor(0.0),
        longitude=torch.tensor(0.0),
        altitude=torch.tensor(0.0))

    # Create spacecraft state (with velocity) - Directly above ground station
    spacecraft_states = SpaceCraftStateDict({
        'position_BN_N':
        torch.tensor([6378.137e3 + 100e3, 0.0, 0.0]),  # 100km directly above
        'velocity_BN_N':
        torch.tensor([0.0, 7.5e3, 0.0]),  # 7.5 km/s velocity
        'position_CN_N':
        torch.tensor([6378.137e3 + 100e3, 0.0, 0.0]),
        'velocity_CN_N':
        torch.tensor([0.0, 7.5e3, 0.0]),
        'sigma_BN':
        torch.zeros(3),
        'omega_BN_B':
        torch.zeros(3),
        'omegaDot_BN_B':
        torch.zeros(3),
    })

    # Initialize state dictionary
    state_dict = ground_location.reset()

    # Run calculation
    result = ground_location(state_dict, spacecraft_states, planet_state,
                             position_LP_P_init,
                             direction_cosine_matrix_Planet2Location)

    # Get results
    access_states = result['access_states']

    # Verify results
    accuracy = 1e-6

    # Check basic access information
    print(
        f"Elevation: {access_states['elevation'].item() * 180 / torch.pi:.2f} degrees"
    )
    print(
        f"Azimuth: {access_states['azimuth'].item() * 180 / torch.pi:.2f} degrees"
    )
    print(f"Range: {access_states['slant_range'].item() / 1000:.2f} km")
    print(f"Range rate: {access_states['range_dot'].item():.2f} m/s")

    assert access_states['has_access'] == 1, "Should have access"
    assert access_states['slant_range'] > 0, "Range should be positive"
    assert access_states['elevation'] > 0, "Elevation should be positive"
    assert access_states['azimuth'] >= -torch.pi and access_states[
        'azimuth'] <= torch.pi, "Azimuth should be in valid range"

    # Check rate information
    assert 'range_dot' in access_states, "Range rate should be computed"
    assert 'azimuth_dot' in access_states, "Azimuth rate should be computed"
    assert 'elevation_dot' in access_states, "Elevation rate should be computed"

    print("✅ Az/El/Range rates test passed!")


def test_spherical_planet():
    """
    Test PCPF2LLA for spherical planet (Earth)
    """
    ground_location = GroundLocation(
        planet_radius=6378.137e3,  # Earth equatorial radius [m]
        minimum_elevation=10.0 * torch.pi / 180.0,
        timer=Timer(dt=1.0))

    # Test point at equator, 100km altitude
    test_position = torch.tensor([6478.137e3, 0.0, 0.0])  # 100km above equator

    lla = ground_location.PCPF2LLA(test_position,
                                   ground_location.planet_radius)

    expected_latitude = torch.tensor(0.0)  # Equator
    expected_longitude = torch.tensor(0.0)  # Prime meridian
    expected_altitude = torch.tensor(100e3)  # 100km

    assert torch.allclose(lla[0], expected_latitude,
                          atol=1e-6), "Latitude mismatch"
    assert torch.allclose(lla[1], expected_longitude,
                          atol=1e-6), "Longitude mismatch"
    assert torch.allclose(lla[2], expected_altitude,
                          atol=1.0), "Altitude mismatch"

    print("✅ Spherical planet test passed!")


def test_round_trip_conversion():
    """
    Test round-trip conversion: LLA -> PCPF -> LLA
    """
    ground_location = GroundLocation(
        planet_radius=6378.137e3,  # Earth equatorial radius [m]
        minimum_elevation=10.0 * torch.pi / 180.0,
        timer=Timer(dt=1.0))

    # Test various locations
    test_cases = [
        (0.0, 0.0, 100e3),  # Equator, prime meridian, 100km
        (45.0 * torch.pi / 180.0, 90.0 * torch.pi / 180.0,
         200e3),  # 45°N, 90°E, 200km
        (-30.0 * torch.pi / 180.0, -120.0 * torch.pi / 180.0,
         50e3),  # 30°S, 120°W, 50km
    ]

    for lat, lon, alt in test_cases:
        # LLA to PCPF
        lla_input = torch.stack(
            [torch.tensor(lat),
             torch.tensor(lon),
             torch.tensor(alt)])
        pcpf = ground_location.LLA2PCPF(lla_input,
                                        ground_location.planet_radius)

        # PCPF back to LLA
        lla_output = ground_location.PCPF2LLA(pcpf,
                                              ground_location.planet_radius)

        # Check round-trip accuracy
        assert torch.allclose(
            lla_input, lla_output, atol=1e-6
        ), f"Round-trip conversion failed for lat={lat}, lon={lon}, alt={alt}"

    print("✅ Round-trip conversion test passed!")


def test_ground_location_functionality():
    """
    Test ground location functionality with spherical planet
    """
    # Create planet state
    planet_state = Ephemeris({
        'position_vector': torch.zeros(3),
        'velocity_vector': torch.zeros(3),
        'J2000_2_planet_fixed': torch.eye(3),
        'J2000_2_planet_fixed_dot': torch.zeros(3, 3),
    })

    # Initialize ground station with spherical Earth
    ground_location = GroundLocation(
        planet_radius=6378.137e3,  # Earth equatorial radius [m]
        maximum_range=200e3,
        minimum_elevation=10.0 * torch.pi / 180.0,
        timer=Timer(dt=1.0))

    # Set ground station location (equator, 0 degrees longitude)
    position_LP_P_init, direction_cosine_matrix_Planet2Location = ground_location.specify_location(
        latitude=torch.tensor(0.0),
        longitude=torch.tensor(0.0),
        altitude=torch.tensor(0.0))

    # Create spacecraft state
    spacecraft_states = SpaceCraftStateDict({
        'position_BN_N':
        torch.tensor([6378.137e3 + 100e3, 0.0, 0.0]),
        'velocity_BN_N':
        torch.zeros(3),
        'position_CN_N':
        torch.tensor([6378.137e3 + 100e3, 0.0, 0.0]),
        'velocity_CN_N':
        torch.zeros(3),
        'sigma_BN':
        torch.zeros(3),
        'omega_BN_B':
        torch.zeros(3),
        'omegaDot_BN_B':
        torch.zeros(3),
    })

    # Initialize state dictionary
    state_dict = ground_location.reset()

    # Run calculation
    result = ground_location(state_dict, spacecraft_states, planet_state,
                             position_LP_P_init,
                             direction_cosine_matrix_Planet2Location)

    # Get results
    access_states = result['access_states']

    # Verify results
    expected_range = torch.tensor(100e3)
    expected_elevation = torch.tensor(torch.pi / 2)  # 90 degrees
    expected_access = torch.tensor(1, dtype=torch.uint8)

    assert torch.allclose(access_states['slant_range'],
                          expected_range,
                          atol=1.0), "Range mismatch"
    assert torch.allclose(access_states['elevation'],
                          expected_elevation,
                          atol=1e-6), "Elevation mismatch"
    assert torch.allclose(access_states['has_access'],
                          expected_access), "Access mismatch"

    print("✅ Ground location functionality test passed!")


if __name__ == "__main__":
    raise ValueError("Not implemented")
