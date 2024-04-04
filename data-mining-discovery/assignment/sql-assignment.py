# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:05:29 2024

@author: kthat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def age_calculate(df):
    """
    This method to calculate age of person based on today date
    """
    current_date = pd.Timestamp('now')
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'])
    print(df.head())
    df['Age'] = (current_date - df['DateOfBirth']) / np.timedelta64(1, 'Y')
    df['Age'] = df['Age'].round(0)
    print(df.head())
    return df



"""
    Prepare patients tables data 
    With all Nominal, Ordinal, Interval and Ratio type of attributes
"""
# Number of samples
npatients = 1000

# Nominal data: Names, Ids
names = pd.read_csv('1000_names.csv', delimiter=',',
                    dtype=str)
first_names = names['First Name']
last_names = names['Last Name']
email = first_names + '_' + last_names + '@' + 'email.com'

patientid = [f'PT{str(i).zfill(4)}' for i in range(1, 1001)]
patient_profile_id = np.unique(patientid)
patient_record_id = np.random.choice(patient_profile_id, npatients)

# Nominal data: Address
postcodes = [f'AL{str(i).zfill(2)}' for i in range(1, 11)]
postcode_data = np.random.choice(postcodes, npatients)

# Ordinal data: Blood group, Gender
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'O+', 'O-', 'AB-']
blood_group_data = np.random.choice(
    blood_groups, npatients, p=[0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1])

gender = ['Female', 'Male']
gender_data = np.random.choice(gender, npatients)

# Interval data: Age of person
birth_year = np.random.randint(1925, 2020, npatients)  # no new borns
birth_month = np.random.randint(1, 13, npatients)
birth_day = np.random.randint(1, 29, npatients)
date_of_birth = [f'{birth_year[i]}-{str(birth_month[i]).zfill(2)}-'
                 f'{str(birth_day[i]).zfill(2)}' for i in range(npatients)]

# Ratio data: Height in cm
height_data = np.random.lognormal(
    mean=5, sigma=0.25, size=npatients).astype(int)

# Ratio data: Weight in Kg
weight = np.random.randint(15, 200, npatients)  # in kg
weight_data = np.random.lognormal(
    mean=4, sigma=0.5, size=npatients).astype(int)

# Ordinal data: Alergic
alergic = ['drug allergy', 'food allergy', 'eczema',
           'hives', 'asthma', 'latex', 'other', '']
alergic_data = np.random.choice(alergic, npatients)

# Ordinal data: Desease category
deseased = ['YES', 'NO', '']
deseased_data = np.random.choice(deseased, npatients, p=[0.6, 0.3, 0.1])

# Interval data: Record date
reg_year = np.random.randint(2020, 2023, npatients)
reg_month = np.random.randint(1, 13, npatients)
reg_day = np.random.randint(1, 29, npatients)
history_record_date = [f'{reg_year[i]}-{str(reg_month[i]).zfill(2)}-'
                       f'{str(reg_day[i]).zfill(2)}' for i in range(npatients)]

# comment about medical condition by doctor
clinical_finding = ['to be added by doctor', '']
clinical_finding_data = np.random.choice(clinical_finding, npatients)

pulse_rate = np.random.randint(60, 120, npatients)  # in bpm (beats per minute)

body_temperature = np.round(np.random.uniform(
    95, 105, npatients), 2)  # in Farahniet


""" 
    Prepare doctors table data 
"""
ndoctors = 100
doctors_names = pd.read_csv(' ,.''doctors-name.csv', delimiter=',',
                            dtype=str)
firstname = doctors_names['FirstName']
lastname = doctors_names['LastName']
doctors_names['FullName'] = firstname + ' ' + lastname
fullname = doctors_names['FullName']

doctorid = [f'DR{str(i).zfill(2)}' for i in range(1, 101)]
doctor_id = np.unique(doctorid)

phone = np.random.randint(730000000, 760000000, size=ndoctors, dtype=int)

spcialized = ['General Physician', 'Asthetic Physician', 'Allergy Specialist', 'Anaesthetist', 'Cardiologist', 'Child Psychologist', 'Dermatologist', 'Fertility Consultant', 'Nutritionist',
              'Neuro Physician', 'Paediatric', 'General Surgeon', 'Orthodontist', 'Gynecologist', 'Haematologist', 'Eye Surgeon', 'Dental Surgeon', 'Neurologist', 'Family Physician', 'Paediatric']
spcialized_data = np.random.choice(spcialized, ndoctors)



""" 
    Prepare appointment table data for two months period 
"""
nappointments = 100

# Interval data: appointment date
appointment_year = np.random.randint(2024, 2025, npatients)
appointment_month = np.random.randint(1, 3, npatients)
appointment_day = np.random.randint(1, 29, npatients)
appointment_date = [f'{appointment_year[i]}-{str(appointment_month[i]).zfill(2)}-'
                    f'{str(appointment_day[i]).zfill(2)}' for i in range(nappointments)]

appointmentid = [f'{str(i).zfill(2)}' for i in range(1, 11)]
appointment_id = np.random.choice(appointmentid, nappointments)

roomno = [f'R{str(i).zfill(2)}' for i in range(1, 11)]
room_no = np.random.choice(roomno, nappointments)

app_time = [f'{str(i).zfill(2)}:00' for i in range(8, 22)]
appointment_time = np.random.choice(app_time, nappointments)

appointed_patient_id = np.random.choice(patient_profile_id, nappointments)

appointed_doctor_id = np.random.choice(doctor_id, nappointments)


""" 
    Create DataFrames 
    Write into csv file 
    --------------------
"""
df_patient_profile = pd.DataFrame({
    'Id': patient_profile_id,
    'FirstName': first_names,
    'LastName': last_names,
    'Email': email,
    'DateOfBirth': date_of_birth,
    'Gender': gender_data,
    'BloodGroup': blood_group_data,
    'PostCode': postcode_data
})

df_patient_history = pd.DataFrame({
    'PatientId': patient_record_id,
    'Height': height_data,
    'Weight': weight_data,
    'AlergicType': alergic_data,
    'RecordDate': history_record_date,
    'Deseased': deseased_data,
    'PulseRate': pulse_rate,
    'Temperature': body_temperature,
    'ClinicalFindings': clinical_finding_data
})

df_doctors_profile = pd.DataFrame({
    'DoctorId': doctor_id,
    'FullName': fullname,
    'PhoneNo': phone,
    'Specialization': spcialized_data
})

df_appointments = pd.DataFrame({
    'AppointmentId': appointment_id,
    'AppointmentDate': appointment_date,
    'AppointmentTime': appointment_time,
    'RoomNo': room_no,
    'Doctor': appointed_doctor_id,
    'Patient': appointed_patient_id
})

print(df_appointments.head())

df_patient_profile_new = age_calculate(df_patient_profile)
print(df_patient_profile_new.head())


df_patient_profile_new.to_csv(
    'C:\/Users\/kthat\/OneDrive\/MODULES\/DM\/PatientsProfile.csv')
df_patient_history.to_csv(
    'C:\/Users\/kthat\/OneDrive\/MODULES\/DM\/PatientsHistory.csv')
df_doctors_profile.to_csv(
    'C:\/Users\/kthat\/OneDrive\/MODULES\/DM\/DoctorsProfile.csv')
df_appointments.to_csv(
    'C:\/Users\/kthat\/OneDrive\/MODULES\/DM\/Appointments1.csv')
