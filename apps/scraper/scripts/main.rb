load "scripts/scraper_and_saver.rb"

# puts "Enter Repository url: "
# repo_url = gets
#
# puts "Enter Pull Request number: "
# pr_number = gets

# pr_url = "#{repo_url.strip}/pull/#{pr_number.strip}"

params = { pr_url: "#{ARGV[0].strip}/pull/#{ARGV[1].strip}", project_dir: ARGV[2] }

scraper_and_saver params
