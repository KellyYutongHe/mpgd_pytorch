# faceid
python main.py -i wo -s faceid --doc celeba_hq --timesteps 50 --rho_scale 0.015 --stop 100 --batch_size 1 --eta 0.5 --ref_path ./images/294.jpg --repeat 1
python main.py -i ae -s faceid --doc celeba_hq --timesteps 50 --rho_scale 0.015 --stop 100 --batch_size 1 --eta 0.5 --ref_path ./images/294.jpg --repeat 1 --ldm_start 500 --ldm_end 300
python main.py -i z -s faceid --doc celeba_hq --timesteps 50 --rho_scale 0.015 --stop 100 --batch_size 1 --eta 0.5 --ref_path ./images/294.jpg --repeat 1 --ldm_start 300 --ldm_end 100

# clip
python main.py -i wo -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 1.5 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a person wearing red lipstick" --repeat 1 --repeat_start 500 --repeat_end 200
python main.py -i ae -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 1 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a person wearing red lipstick" --repeat 1 --repeat_start 500 --repeat_end 200 --ldm_start 500 --ldm_end 300
python main.py -i z -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 2 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a person wearing red lipstick" --repeat 1 --repeat_start 500 --repeat_end 200 --ldm_start 400 --ldm_end 200
python main.py -i z -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 2.5 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a person with blond hair" --repeat 3 --repeat_start 700 --repeat_end 400 --ldm_start 400 --ldm_end 200
python main.py -i z -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 2.5 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a man" --repeat 5 --repeat_start 700 --repeat_end 400 --ldm_start 400 --ldm_end 200